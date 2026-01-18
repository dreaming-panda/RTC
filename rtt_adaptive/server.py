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


ICE_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
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
    user_actions: List[Action]  # every frame carries user_actions


class BatchEngine:
    """
    蓝色正弦波纹：blank action 不改变颜色；user action 触发颜色变化
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
        if any(a.is_user and a.id for a in actions):
            self.color_phase += 0.8

    def infer(self, infer_id: int, actions: List[Action], anim_fps: float, frames_per_infer: int) -> List[np.ndarray]:
        self._apply_actions(actions)
        user_texts = [a.text for a in actions if a.is_user and a.id]
        actions_text = " | ".join(user_texts) if user_texts else "(none)"

        cp = self.color_phase
        base_r = 0.20 + 0.20 * np.sin(cp + 0.0)
        base_g = 0.25 + 0.25 * np.sin(cp + 2.1)
        base_b = 0.75 + 0.20 * np.sin(cp + 4.2)

        frames: List[np.ndarray] = []
        for _ in range(frames_per_infer):
            t = (self.global_frame_seq / max(anim_fps, 1e-6)) * self.speed
            phase = self.freq * self.rr * (2.0 * np.pi) - t
            wave = (np.sin(phase) * 0.5 + 0.5).astype(np.float32)

            r = wave * base_r
            g = wave * base_g
            b = wave * base_b
            img = np.stack([r, g, b], axis=-1)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

            if cv2 is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"infer_id: {infer_id}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(bgr, f"global_frame: {self.global_frame_seq}", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(bgr, f"user_actions: {actions_text[:70]}", (12, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(bgr, f"frames_per_infer: {frames_per_infer}", (12, 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            frames.append(img)
            self.global_frame_seq += 1

        return frames


class AdaptiveFpsController:
    """
    不用 PID：带滞回 + 平滑(EMA) 的自适应 fps 控制器。

    direction:
      +1: q 高 -> fps 增加（用于 send）
      -1: q 高 -> fps 减少（用于 infer）
    """
    def __init__(
        self,
        base_fps: float,
        min_fps: float,
        max_fps: float,
        target_q: int = 8,
        hyst: int = 2,
        up_gain: float = 0.10,
        down_gain: float = 0.08,
        relax_gain: float = 0.03,
        ema: float = 0.18,
        direction: int = +1,
    ):
        self.base_fps = float(base_fps)
        self.min_fps = float(min_fps)
        self.max_fps = float(max_fps)
        self.target_q = int(target_q)
        self.hyst = int(hyst)
        self.up_gain = float(up_gain)
        self.down_gain = float(down_gain)
        self.relax_gain = float(relax_gain)
        self.ema = float(ema)
        self.direction = +1 if direction >= 0 else -1

        self.fps = float(base_fps)

    def update(self, qsize: int) -> float:
        q = int(qsize)
        desired = self.fps

        hi = self.target_q + self.hyst
        lo = self.target_q - self.hyst

        if q > hi:
            err = q - hi
            if self.direction > 0:
                desired = self.fps * (1.0 + self.up_gain) + 0.12 * err
            else:
                desired = self.fps * (1.0 - self.down_gain) - 0.06 * err
        elif q < lo:
            err = lo - q
            if self.direction > 0:
                desired = self.fps * (1.0 - self.down_gain) - 0.06 * err
            else:
                desired = self.fps * (1.0 + self.up_gain) + 0.12 * err
        else:
            desired = self.fps + (self.base_fps - self.fps) * self.relax_gain

        desired = max(self.min_fps, min(self.max_fps, desired))
        self.fps = (1.0 - self.ema) * self.fps + self.ema * desired
        self.fps = max(self.min_fps, min(self.max_fps, self.fps))
        return self.fps


class Pipeline:
    def __init__(
        self,
        engine: BatchEngine,
        infer_fps_ctrl: AdaptiveFpsController,
        actions_per_infer: int,
        frames_per_infer: int,
        sim_infer_ms: int,
        frame_queue_max: int,
        action_collect_timeout_sec: float,
        infer_in_thread: bool = True,
    ):
        self.engine = engine
        self.infer_fps_ctrl = infer_fps_ctrl
        self.actions_per_infer = int(actions_per_infer)
        self.frames_per_infer = int(frames_per_infer)
        self.sim_infer_ms = int(sim_infer_ms)
        self.action_collect_timeout_sec = float(action_collect_timeout_sec)
        self.infer_in_thread = bool(infer_in_thread)

        self.action_q: asyncio.Queue[Action] = asyncio.Queue()
        self.frame_q: asyncio.Queue[FrameItem] = asyncio.Queue(maxsize=frame_queue_max)
        self.datachannels: Set = set()

        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._infer_id = 0

        self._last_frame: Optional[FrameItem] = None

        self._last_infer_ms: float = 0.0
        self._last_q_before: int = 0
        self._last_q_after: int = 0
        self._last_infer_fps: float = float(infer_fps_ctrl.base_fps)

        self._last_infer_stats_push = 0.0

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

        while len(actions) < self.actions_per_infer:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                a = await asyncio.wait_for(self.action_q.get(), timeout=remaining)
                actions.append(a)
            except asyncio.TimeoutError:
                break

        while len(actions) < self.actions_per_infer:
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
            q_now = self.frame_q.qsize()
            infer_fps = self.infer_fps_ctrl.update(q_now)
            self._last_infer_fps = infer_fps

            infer_period = self.frames_per_infer / max(infer_fps, 1e-6)

            now = time.time()
            if now < next_tick:
                await asyncio.sleep(min(0.01, next_tick - now))
                continue
            next_tick = now + infer_period

            actions = await self._collect_actions_for_one_infer()
            user_actions = [a for a in actions if a.is_user and a.id]

            self._infer_id += 1
            infer_id = self._infer_id

            # infer_start (user only)
            t_start_ms = int(time.time() * 1000)
            for a in user_actions:
                self.broadcast({
                    "type": "infer_start",
                    "id": a.id,
                    "action": a.text,
                    "infer_id": infer_id,
                    "t_server_ms": t_start_ms,
                })

            q_before = self.frame_q.qsize()
            t0 = time.time()

            if self.sim_infer_ms > 0:
                await asyncio.sleep(self.sim_infer_ms / 1000.0)

            if self.infer_in_thread:
                frames = await asyncio.to_thread(
                    self.engine.infer, infer_id, actions, infer_fps, self.frames_per_infer
                )
            else:
                frames = self.engine.infer(infer_id, actions, infer_fps, self.frames_per_infer)

            t1 = time.time()
            q_after = self.frame_q.qsize()

            self._last_q_before = int(q_before)
            self._last_q_after = int(q_after)

            infer_ms = (t1 - t0) * 1000.0
            self._last_infer_ms = infer_ms

            print(f"[INFER] infer_id={infer_id:4d} dt={infer_ms:7.1f}ms "
                  f"q_before={q_before:3d} q_after={q_after:3d} infer_fps={infer_fps:5.2f} "
                  f"actions_per_infer={self.actions_per_infer} frames_per_infer={self.frames_per_infer}")

            # infer_end (user only)
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

            # enqueue frames; drop-oldest on overflow
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
                    user_actions=user_actions,
                ))

            # push infer_stats (low rate)
            t = time.time()
            if t - self._last_infer_stats_push > 0.25:
                self._last_infer_stats_push = t
                self.broadcast({
                    "type": "infer_stats",
                    "infer_id": infer_id,
                    "infer_ms": round(infer_ms, 2),
                    "q_before": int(q_before),
                    "q_after": int(q_after),
                    "frame_q_after_enqueue": self.frame_q.qsize(),
                    "infer_fps": round(infer_fps, 3),
                    "infer_period_ms": round(infer_period * 1000.0, 2),
                    "actions_per_infer": self.actions_per_infer,
                    "frames_per_infer": self.frames_per_infer,
                    "ts_ms": int(time.time() * 1000),
                })


class AdaptiveSendVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, pipeline: Pipeline, send_fps_ctrl: AdaptiveFpsController):
        super().__init__()
        self.pipeline = pipeline
        self.send_fps_ctrl = send_fps_ctrl

        self._last_send_t: Optional[float] = None
        self._applied_sent_for_infer: Set[int] = set()

        self._last_log_t = 0.0
        self._last_stats_push_t = 0.0

        self._last_send_fps: float = float(send_fps_ctrl.base_fps)

    async def recv(self):
        qsize = self.pipeline.frame_q.qsize()
        send_fps = self.send_fps_ctrl.update(qsize)
        self._last_send_fps = send_fps
        interval = 1.0 / max(send_fps, 1e-6)

        now = time.time()
        if self._last_send_t is not None:
            delay = interval - (now - self._last_send_t)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last_send_t = time.time()

        item = await self.pipeline.get_frame_item()
        q2 = self.pipeline.frame_q.qsize()

        # applied: when this infer's first dequeued frame is about to be sent
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

        t = time.time()
        if t - self._last_log_t > 0.5:
            self._last_log_t = t
            print(f"[FRAME_Q] size={q2:3d} infer_id={item.infer_id:4d} idx={item.frame_index:2d} "
                  f"send_fps={send_fps:5.2f} infer_fps={self.pipeline._last_infer_fps:5.2f} "
                  f"q_before/after={self.pipeline._last_q_before}/{self.pipeline._last_q_after}")

        if t - self._last_stats_push_t > 0.5:
            self._last_stats_push_t = t
            est_queue_ms = (q2 / max(send_fps, 1e-6)) * 1000.0
            self.pipeline.broadcast({
                "type": "stats",
                "frame_q": q2,
                "infer_id": item.infer_id,
                "frame_index": item.frame_index,

                "send_fps": round(send_fps, 3),
                "send_base_fps": round(self.send_fps_ctrl.base_fps, 3),
                "send_range": [round(self.send_fps_ctrl.min_fps, 2), round(self.send_fps_ctrl.max_fps, 2)],

                "infer_fps": round(self.pipeline._last_infer_fps, 3),
                "infer_base_fps": round(self.pipeline.infer_fps_ctrl.base_fps, 3),
                "infer_range": [round(self.pipeline.infer_fps_ctrl.min_fps, 2), round(self.pipeline.infer_fps_ctrl.max_fps, 2)],

                "actions_per_infer": self.pipeline.actions_per_infer,
                "frames_per_infer": self.pipeline.frames_per_infer,

                "target_q": self.send_fps_ctrl.target_q,
                "est_queue_ms": round(est_queue_ms, 1),

                "last_infer_ms": round(self.pipeline._last_infer_ms, 2),
                "last_q_before": self.pipeline._last_q_before,
                "last_q_after": self.pipeline._last_q_after,
                "ts_ms": int(time.time() * 1000),
            })

        frame = av.VideoFrame.from_ndarray(item.img, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


pcs = set()


def make_app(pipeline: Pipeline, send_fps_ctrl: AdaptiveFpsController):
    app = web.Application()

    async def index(request):
        here = os.path.dirname(os.path.abspath(__file__))
        return web.FileResponse(os.path.join(here, "client.html"))

    async def offer(request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection(ICE_CONFIG)
        pcs.add(pc)

        pc.addTrack(AdaptiveSendVideoTrack(pipeline, send_fps_ctrl))

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

                # ACK
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
    p.add_argument("--port", type=int, default=8090)

    # NEW: make these configurable
    p.add_argument("--actions-per-infer", type=int, default=1)
    p.add_argument("--frames-per-infer", type=int, default=4)

    # inference simulation
    p.add_argument("--sim-infer-ms", type=int, default=120)
    p.add_argument("--frame-queue-max", type=int, default=128)
    p.add_argument("--action-collect-timeout-ms", type=int, default=30)
    p.add_argument("--infer-in-thread", action="store_true", default=True)

    # shared queue target
    p.add_argument("--target-q", type=int, default=8)
    p.add_argument("--hyst", type=int, default=2)

    # adaptive infer fps (production): q high -> infer_fps down
    p.add_argument("--infer-base-fps", type=float, default=20.0)
    p.add_argument("--infer-min-fps", type=float, default=16.0)
    p.add_argument("--infer-max-fps", type=float, default=22.0)

    # adaptive send fps (consumption): q high -> send_fps up
    p.add_argument("--send-base-fps", type=float, default=20.0)
    p.add_argument("--send-min-fps", type=float, default=16.0)
    p.add_argument("--send-max-fps", type=float, default=22.0)

    args = p.parse_args()

    if args.actions_per_infer < 1:
        raise SystemExit("--actions-per-infer must be >= 1")
    if args.frames_per_infer < 1:
        raise SystemExit("--frames-per-infer must be >= 1")

    engine = BatchEngine(w=640, h=360)

    infer_fps_ctrl = AdaptiveFpsController(
        base_fps=args.infer_base_fps,
        min_fps=args.infer_min_fps,
        max_fps=args.infer_max_fps,
        target_q=args.target_q,
        hyst=args.hyst,
        direction=-1,  # q high -> infer fps down
        ema=0.18,
    )

    send_fps_ctrl = AdaptiveFpsController(
        base_fps=args.send_base_fps,
        min_fps=args.send_min_fps,
        max_fps=args.send_max_fps,
        target_q=args.target_q,
        hyst=args.hyst,
        direction=+1,  # q high -> send fps up
        ema=0.18,
    )

    pipeline = Pipeline(
        engine=engine,
        infer_fps_ctrl=infer_fps_ctrl,
        actions_per_infer=args.actions_per_infer,
        frames_per_infer=args.frames_per_infer,
        sim_infer_ms=args.sim_infer_ms,
        frame_queue_max=args.frame_queue_max,
        action_collect_timeout_sec=args.action_collect_timeout_ms / 1000.0,
        infer_in_thread=args.infer_in_thread,
    )

    print(f"[CFG] actions_per_infer={args.actions_per_infer} frames_per_infer={args.frames_per_infer}")
    print(f"[TARGET] target_q={args.target_q} hyst={args.hyst}")
    print(f"[INFER_FPS] base={infer_fps_ctrl.base_fps} range=[{infer_fps_ctrl.min_fps},{infer_fps_ctrl.max_fps}] (q high -> down)")
    print(f"[SEND_FPS ] base={send_fps_ctrl.base_fps} range=[{send_fps_ctrl.min_fps},{send_fps_ctrl.max_fps}] (q high -> up)")
    print(f"[INFER] sim_infer_ms={args.sim_infer_ms} infer_in_thread={args.infer_in_thread} frame_queue_max={args.frame_queue_max}")

    web.run_app(make_app(pipeline, send_fps_ctrl), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
