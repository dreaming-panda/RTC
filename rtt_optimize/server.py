import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Dict

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


ICE_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        # 如云上仍 failed，需要 TURN
        # RTCIceServer(urls=["turn:YOUR_TURN:3478?transport=udp"], username="u", credential="p"),
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
    frame_index: int
    # 为了鲁棒：每一帧都带 user_actions，避免队列很小/丢帧导致拿不到“第0帧”
    user_actions: List[Action]


class BatchEngine:
    """
    蓝色正弦波纹。
    blank action 不改变颜色状态；
    user action 出现则推进 color_phase，导致整体颜色变化。
    """
    def __init__(self, w=640, h=360):
        self.w = w
        self.h = h
        self.global_frame_seq = 0

        self.color_phase = 0.0
        self.speed = 1.0
        self.freq = 6.0

        x = np.linspace(-1.0, 1.0, self.w, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, self.h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        self.rr = np.sqrt(xx * xx + yy * yy).astype(np.float32)

    def _apply_actions(self, actions: List[Action]):
        user_cnt = sum(1 for a in actions if a.is_user and a.id)
        if user_cnt > 0:
            # 一次 batch 只要出现 user action，就变一次色
            self.color_phase += 0.8

    def infer(self, infer_id: int, actions: List[Action], output_fps: float) -> List[np.ndarray]:
        self._apply_actions(actions)

        user_texts = [a.text for a in actions if a.is_user and a.id]
        actions_text = " | ".join(user_texts) if user_texts else "(none)"

        frames: List[np.ndarray] = []
        cp = self.color_phase
        base_r = 0.20 + 0.20 * np.sin(cp + 0.0)
        base_g = 0.25 + 0.25 * np.sin(cp + 2.1)
        base_b = 0.75 + 0.20 * np.sin(cp + 4.2)

        for k in range(FRAMES_PER_INFER):
            t = (self.global_frame_seq / max(output_fps, 1e-6)) * self.speed
            phase = self.freq * self.rr * (2.0 * np.pi) - t
            wave = (np.sin(phase) * 0.5 + 0.5).astype(np.float32)

            r = (wave * base_r)
            g = (wave * base_g)
            b = (wave * base_b)

            img = np.stack([r, g, b], axis=-1)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

            if cv2 is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"infer_id: {infer_id}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(bgr, f"global_frame: {self.global_frame_seq}", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(bgr, f"user_actions: {actions_text[:70]}", (12, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            frames.append(img)
            self.global_frame_seq += 1

        return frames


class Pipeline:
    """
    - 周期触发：每次 infer 需要 4 个 action，不够用 blank 补齐
    - 模拟 infer：await asyncio.sleep(sim_infer_ms/1000)
    - 事件追踪（仅对 user action）：
        ack：收到即发（在 on_message 里）
        infer_start：开始 infer 前发
        infer_end：infer 完成后发（含 infer_ms）
        applied：当该 infer 的第一帧真正出队准备发送时发（在 VideoTrack 触发）
    - stats：每 0.5s 推送 frame_q size 等
    """
    def __init__(
        self,
        engine: BatchEngine,
        output_fps: float,
        sim_infer_ms: int,
        frame_queue_max: int,
        action_collect_timeout_sec: float,
    ):
        self.engine = engine
        self.output_fps = float(output_fps)
        self.sim_infer_ms = int(sim_infer_ms)
        self.action_collect_timeout_sec = float(action_collect_timeout_sec)

        self.action_q: asyncio.Queue[Action] = asyncio.Queue()
        self.frame_q: asyncio.Queue[FrameItem] = asyncio.Queue(maxsize=frame_queue_max)
        self.datachannels: Set = set()

        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._infer_id = 0

        self._last_frame: Optional[FrameItem] = None
        self._last_infer_ms: float = 0.0

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
        deadline = time.time() + self.action_collect_timeout_sec

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
        infer_period_sec = FRAMES_PER_INFER / max(self.output_fps, 1e-6)
        next_tick = time.time()

        while not self._stop.is_set():
            now = time.time()
            if now < next_tick:
                await asyncio.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + infer_period_sec

            actions = await self._collect_actions_for_one_infer()
            user_actions = [a for a in actions if a.is_user and a.id]

            self._infer_id += 1
            infer_id = self._infer_id

            # infer_start（只对 user）
            t_start_ms = int(time.time() * 1000)
            for a in user_actions:
                self.broadcast({
                    "type": "infer_start",
                    "id": a.id,
                    "action": a.text,
                    "infer_id": infer_id,
                    "t_server_ms": t_start_ms,
                })

            # 模拟 infer cost（语义正确）
            infer_t0 = time.time()
            if self.sim_infer_ms > 0:
                await asyncio.sleep(self.sim_infer_ms / 1000.0)

            # 生成 frames（你换成真实模型即可）
            frames = self.engine.infer(infer_id=infer_id, actions=actions, output_fps=self.output_fps)
            infer_t1 = time.time()
            infer_ms = (infer_t1 - infer_t0) * 1000.0
            self._last_infer_ms = infer_ms

            # infer_end（只对 user），带 infer_ms
            t_end_ms = int(time.time() * 1000)
            for a in user_actions:
                self.broadcast({
                    "type": "infer_end",
                    "id": a.id,
                    "action": a.text,
                    "infer_id": infer_id,
                    "infer_ms": round(infer_ms, 2),
                    "t_server_ms": t_end_ms,
                })

            # 入队 16 帧：队列满则丢“最旧”(drop-oldest)
            for idx, img in enumerate(frames):
                if self.frame_q.full():
                    try:
                        _ = self.frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass

                await self.frame_q.put(FrameItem(
                    img=img,
                    infer_id=infer_id,
                    frame_index=idx,
                    user_actions=user_actions,  # 每帧都带，鲁棒
                ))

            # 推一个 infer 统计（方便你对照）
            self.broadcast({
                "type": "infer_stats",
                "infer_id": infer_id,
                "infer_ms": round(infer_ms, 2),
                "frame_q_after": self.frame_q.qsize(),
                "ts_ms": int(time.time() * 1000),
            })


class PipelineVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, pipeline: Pipeline):
        super().__init__()
        self.pipeline = pipeline
        self._interval = 1.0 / max(self.pipeline.output_fps, 1e-6)
        self._last = None

        self._applied_sent_for_infer: Set[int] = set()
        self._last_log_t = 0.0
        self._last_stats_push_t = 0.0

    async def recv(self):
        # fps 节流（模拟发送端固定帧率输出）
        now = time.time()
        if self._last is not None:
            delay = self._interval - (now - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last = time.time()

        item = await self.pipeline.get_frame_item()
        qsize = self.pipeline.frame_q.qsize()

        # applied：当该 infer 的“第一帧开始出队并准备发送”时发（包含 infer cost + 排队）
        if item.infer_id not in self._applied_sent_for_infer and item.user_actions:
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

        # stdout 打印 frame_q（每 0.5s）
        t = time.time()
        if t - self._last_log_t > 0.5:
            self._last_log_t = t
            print(f"[FRAME_Q] size={qsize:3d} | infer_id={item.infer_id:4d} | idx={item.frame_index:2d}")

        # 推 stats 给前端（每 0.5s）
        if t - self._last_stats_push_t > 0.5:
            self._last_stats_push_t = t
            self.pipeline.broadcast({
                "type": "stats",
                "frame_q": qsize,
                "infer_id": item.infer_id,
                "frame_index": item.frame_index,
                "output_fps": self.pipeline.output_fps,
                "last_infer_ms": round(self.pipeline._last_infer_ms, 2),
                "ts_ms": int(time.time() * 1000),
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

        pc.addTrack(PipelineVideoTrack(pipeline))

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

                # ACK：收到并入队（不包含 infer cost）
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
    p.add_argument("--fps", type=float, default=30.0, help="OUTPUT_FPS")
    p.add_argument("--sim-infer-ms", type=int, default=0, help="simulate infer cost in ms")
    p.add_argument("--frame-queue-max", type=int, default=128)
    p.add_argument("--action-collect-timeout-ms", type=int, default=30)
    args = p.parse_args()

    engine = BatchEngine(w=640, h=360)
    pipeline = Pipeline(
        engine=engine,
        output_fps=args.fps,
        sim_infer_ms=args.sim_infer_ms,
        frame_queue_max=args.frame_queue_max,
        action_collect_timeout_sec=args.action_collect_timeout_ms / 1000.0,
    )

    if args.frame_queue_max < FRAMES_PER_INFER:
        print(f"[WARN] frame_queue_max({args.frame_queue_max}) < FRAMES_PER_INFER({FRAMES_PER_INFER}); "
              f"may drop within a batch. Applied is still robust (user_actions on every frame).")

    web.run_app(make_app(pipeline), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
