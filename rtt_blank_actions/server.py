import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np
import av
from aiohttp import web

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)

try:
    import cv2
except Exception:
    cv2 = None


ACTIONS_PER_INFER = 4
FRAMES_PER_INFER = 16
OUTPUT_FPS = 30

INFER_PERIOD_SEC = FRAMES_PER_INFER / OUTPUT_FPS
ACTION_COLLECT_TIMEOUT_SEC = 0.03
FRAME_QUEUE_MAX = 128

ICE_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        # 需要 TURN 时加这里
    ]
)


async def wait_ice_complete(pc: RTCPeerConnection, timeout: float = 8.0):
    if pc.iceGatheringState == "complete":
        return
    ev = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _():
        if pc.iceGatheringState == "complete":
            ev.set()

    try:
        await asyncio.wait_for(ev.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        print("[WARN] ICE gathering timeout; SDP may miss candidates.")


@dataclass
class Action:
    id: str
    text: str
    t_send_ms: int
    t_recv_ms: int
    is_user: bool


@dataclass
class FrameItem:
    img: np.ndarray
    infer_id: int
    frame_index: int  # 0..15
    user_actions: List[Action]  # only attached on frame_index==0


class BatchEngine:
    def __init__(self, w=640, h=360):
        self.w = w
        self.h = h
        self.global_frame_seq = 0

    def infer(self, infer_id: int, actions: List[Action]) -> List[np.ndarray]:
        actions_text = " | ".join([a.text for a in actions])

        frames: List[np.ndarray] = []
        for _ in range(FRAMES_PER_INFER):
            img = (np.random.rand(self.h, self.w, 3) * 255).astype(np.uint8)

            if cv2 is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"infer_id: {infer_id}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(bgr, f"global_frame: {self.global_frame_seq}", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(bgr, f"actions: {actions_text[:70]}", (12, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            frames.append(img)
            self.global_frame_seq += 1

        return frames


class Pipeline:
    """
    发送 3 类事件（仅对 user action）：
      - infer_start: 该 action 所在 batch 开始 infer
      - infer_end:   该 action 所在 batch infer 结束
      - applied:     该 batch 第 0 帧真正出队准备发送（在 VideoTrack 中触发）
    """
    def __init__(self, engine: BatchEngine):
        self.engine = engine
        self.action_q: asyncio.Queue[Action] = asyncio.Queue()
        self.frame_q: asyncio.Queue[FrameItem] = asyncio.Queue(maxsize=FRAME_QUEUE_MAX)
        self.datachannels: Set = set()

        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._infer_id = 0

        self._last_frame: Optional[FrameItem] = None

    def add_dc(self, dc):
        self.datachannels.add(dc)

    def remove_dc(self, dc):
        self.datachannels.discard(dc)

    def broadcast(self, payload: dict):
        msg = json.dumps(payload)
        dead = []
        for dc in list(self.datachannels):
            try:
                if dc.readyState == "open":
                    dc.send(msg)
                else:
                    dead.append(dc)
            except Exception:
                dead.append(dc)
        for d in dead:
            self.remove_dc(d)

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            await self._task

    async def enqueue_action(self, a: Action):
        await self.action_q.put(a)

    async def _collect_actions_for_one_infer(self) -> List[Action]:
        actions: List[Action] = []

        deadline = time.time() + ACTION_COLLECT_TIMEOUT_SEC
        while len(actions) < ACTIONS_PER_INFER:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                a = await asyncio.wait_for(self.action_q.get(), timeout=remaining)
                actions.append(a)
            except asyncio.TimeoutError:
                break

        while len(actions) < ACTIONS_PER_INFER:
            actions.append(Action(
                id="",
                text="blank",
                t_send_ms=0,
                t_recv_ms=int(time.time() * 1000),
                is_user=False,
            ))
        return actions

    async def get_frame_item(self) -> FrameItem:
        try:
            item = self.frame_q.get_nowait()
            self._last_frame = item
            return item
        except asyncio.QueueEmpty:
            if self._last_frame is not None:
                return self._last_frame
            img = np.zeros((self.engine.h, self.engine.w, 3), dtype=np.uint8)
            item = FrameItem(img=img, infer_id=0, frame_index=0, user_actions=[])
            self._last_frame = item
            return item

    async def _run(self):
        next_tick = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now < next_tick:
                await asyncio.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + INFER_PERIOD_SEC

            actions = await self._collect_actions_for_one_infer()
            user_actions = [a for a in actions if a.is_user and a.id]

            self._infer_id += 1
            infer_id = self._infer_id

            # ✅ infer_start（只对 user action）
            t_infer_start_ms = int(time.time() * 1000)
            for a in user_actions:
                self.broadcast({
                    "type": "infer_start",
                    "id": a.id,
                    "action": a.text,
                    "infer_id": infer_id,
                    "t_server_ms": t_infer_start_ms,
                })

            # ✅ infer（你的真实模型替换这里）
            frames = self.engine.infer(infer_id=infer_id, actions=actions)

            # ✅ infer_end（只对 user action）
            t_infer_end_ms = int(time.time() * 1000)
            for a in user_actions:
                self.broadcast({
                    "type": "infer_end",
                    "id": a.id,
                    "action": a.text,
                    "infer_id": infer_id,
                    "t_server_ms": t_infer_end_ms,
                })

            # 入队 16 帧：仅第0帧携带 user_actions（用于 applied）
            for idx, img in enumerate(frames):
                if self.frame_q.full():
                    try:
                        _ = self.frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                item = FrameItem(
                    img=img,
                    infer_id=infer_id,
                    frame_index=idx,
                    user_actions=(user_actions if idx == 0 else []),
                )
                await self.frame_q.put(item)


class PipelineVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, pipeline: Pipeline, fps: int = OUTPUT_FPS):
        super().__init__()
        self.pipeline = pipeline
        self.fps = fps
        self._interval = 1.0 / float(fps)
        self._last = None
        self._applied_sent_for_infer: Set[int] = set()

    async def recv(self):
        now = time.time()
        if self._last is not None:
            delay = self._interval - (now - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last = time.time()

        item = await self.pipeline.get_frame_item()

        # ✅ applied：该批第0帧真正出队准备发送时才发（包含 infer cost + 排队）
        if (
            item.frame_index == 0
            and item.infer_id not in self._applied_sent_for_infer
            and item.user_actions
        ):
            self._applied_sent_for_infer.add(item.infer_id)
            t_applied_ms = int(time.time() * 1000)
            for a in item.user_actions:
                self.pipeline.broadcast({
                    "type": "applied",
                    "id": a.id,
                    "action": a.text,
                    "infer_id": item.infer_id,
                    "t_server_ms": t_applied_ms,
                })

        frame = av.VideoFrame.from_ndarray(item.img, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


pcs = set()

def make_app(pipeline: Pipeline):
    app = web.Application()

    async def index(request):
        here = os.path.dirname(os.path.abspath(__file__))
        return web.FileResponse(os.path.join(here, "client.html"))

    async def offer(request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection(ICE_CONFIG)
        pcs.add(pc)

        pc.addTrack(PipelineVideoTrack(pipeline, fps=OUTPUT_FPS))

        @pc.on("datachannel")
        def on_datachannel(dc):
            pipeline.add_dc(dc)
            print("DataChannel:", dc.label)

            @dc.on("close")
            def on_close():
                pipeline.remove_dc(dc)

            @dc.on("message")
            def on_message(msg):
                if not isinstance(msg, str):
                    return
                try:
                    data = json.loads(msg)
                except Exception:
                    return

                if data.get("cmd") != "action":
                    return

                action_text = str(data.get("action", "action"))
                action_id = str(data.get("id", ""))
                t_send_ms = int(data.get("t_send_ms", 0))

                a = Action(
                    id=action_id,
                    text=action_text,
                    t_send_ms=t_send_ms,
                    t_recv_ms=int(time.time() * 1000),
                    is_user=True,
                )
                asyncio.create_task(pipeline.enqueue_action(a))

                # ACK：仅表示“收到并入队”
                try:
                    dc.send(json.dumps({
                        "type": "ack",
                        "id": action_id,
                        "action": action_text,
                        "t_send_ms": t_send_ms,
                        "t_server_ms": int(time.time() * 1000),
                    }))
                except Exception:
                    pass

        @pc.on("connectionstatechange")
        async def on_conn():
            print("connectionState:", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                pcs.discard(pc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_ice_complete(pc)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def on_startup(app):
        await pipeline.start()

    async def on_shutdown(app):
        await pipeline.stop()
        await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
        pcs.clear()

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    engine = BatchEngine(w=640, h=360)
    pipeline = Pipeline(engine)

    web.run_app(make_app(pipeline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
