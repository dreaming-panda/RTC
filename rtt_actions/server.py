import argparse
import asyncio
import json
import os
import time
import math
from typing import Optional, Dict

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


# ---------- WebRTC ICE（云服务器常用：STUN；很多网络仍需要 TURN） ----------
ICE_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        # 如果你有 TURN，放在这里（生产推荐）
        # RTCIceServer(urls=["turn:YOUR_TURN:3478?transport=udp"], username="u", credential="p"),
    ]
)

OUTPUT_FPS = 30


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


# ---------- Engine：每次 step 输出一帧（模拟） ----------
class SimpleEngine:
    """
    单步推理：step() -> frame
    - set_action(action_id, action_text) 会让下一帧把 action 叠上去，并标记“已应用”
    """

    def __init__(self, width=640, height=360, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        # 视觉参数（模拟模型）
        self.speed = 1.0
        self.freq = 8.0
        self.color_shift = 0.0

        # 用于叠字的状态
        self.frame_seq = 0
        self.last_action_text = "none"
        self.last_action_id = None

        # “待应用 action”（由 DataChannel 设置，下一次 step 应用）
        self._pending_action_id: Optional[str] = None
        self._pending_action_text: Optional[str] = None

        x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        self._rr = np.sqrt(xx * xx + yy * yy).astype(np.float32)

        self._t0 = time.time()

    def set_action(self, action_id: str, action_text: str):
        self._pending_action_id = action_id
        self._pending_action_text = action_text

    def step(self):
        """
        返回: (rgb_frame_uint8, applied_info_or_none)
        applied_info: {"id":..., "text":..., "frame_seq":...} 表示这一帧第一次体现了该 action
        """
        applied = None

        # 若有 pending action，则“在这一帧应用”
        if self._pending_action_id is not None:
            self.last_action_id = self._pending_action_id
            self.last_action_text = self._pending_action_text or "action"
            applied = {
                "id": self._pending_action_id,
                "text": self.last_action_text,
                "frame_seq": self.frame_seq,
                "t_server_ms": int(time.time() * 1000),
            }
            self._pending_action_id = None
            self._pending_action_text = None

        t = (time.time() - self._t0) * self.speed
        phase = self.freq * self._rr * math.pi * 2.0 - t

        r = (np.sin(phase + self.color_shift) * 0.5 + 0.5)
        g = (np.sin(phase + 2.094 + self.color_shift) * 0.5 + 0.5)
        b = (np.sin(phase + 4.188 + self.color_shift) * 0.5 + 0.5)

        img = np.stack([r, g, b], axis=-1)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

        # 叠字：frame_seq + last_action
        if cv2 is not None:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, f"frame_seq: {self.frame_seq}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(bgr, f"last_action: {self.last_action_text}", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if self.last_action_id:
                cv2.putText(bgr, f"action_id: {self.last_action_id[:8]}", (12, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.frame_seq += 1
        return img, applied


# ---------- VideoTrack：每帧 step 一次，并在 action 应用时通知前端 ----------
class EngineVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, engine: SimpleEngine, fps=OUTPUT_FPS):
        super().__init__()
        self.engine = engine
        self.fps = fps
        self._interval = 1.0 / float(fps)
        self._last = None

        # 由 DataChannel 设置，给“applied”回传用
        self._dc = None

    def set_datachannel(self, dc):
        self._dc = dc

    async def recv(self):
        # 节流
        now = time.time()
        if self._last is not None:
            delay = self._interval - (now - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
        self._last = time.time()

        img, applied = self.engine.step()

        # 如果这一帧应用了 action，就通过 DataChannel 发“applied”
        if applied is not None and self._dc is not None and self._dc.readyState == "open":
            try:
                self._dc.send(json.dumps({
                    "type": "applied",
                    **applied
                }))
            except Exception:
                pass

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


# ---------- aiohttp + aiortc ----------
pcs = set()


def make_app():
    app = web.Application()

    async def index(request):
        here = os.path.dirname(os.path.abspath(__file__))
        return web.FileResponse(os.path.join(here, "client.html"))

    async def offer(request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection(ICE_CONFIG)
        pcs.add(pc)

        engine = SimpleEngine(width=848, height=464, fps=OUTPUT_FPS)
        track = EngineVideoTrack(engine, fps=OUTPUT_FPS)
        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def on_conn():
            print("connectionState:", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                pcs.discard(pc)

        @pc.on("datachannel")
        def on_datachannel(dc):
            print("DataChannel:", dc.label)
            track.set_datachannel(dc)

            @dc.on("message")
            def on_message(msg):
                """
                客户端发送：
                  {"cmd":"action","action":"jump","id":"...","t0":<performance.now>,"t_send_ms":<Date.now>}
                服务器做两件事：
                  1) 立即回 ACK（表示已收到 action）
                  2) 设置 pending action，让下一帧 step() 叠字并发送 applied
                """
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

                # 设置下一帧应用
                engine.set_action(action_id=action_id, action_text=action_text)

                # 立即 ACK（“收到并已排入下一帧应用”）
                ack = {
                    "type": "ack",
                    "id": action_id,
                    "action": action_text,
                    "t_send_ms": t_send_ms,
                    "t_server_ms": int(time.time() * 1000),
                }
                try:
                    dc.send(json.dumps(ack))
                except Exception:
                    pass

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await wait_ice_complete(pc)
        return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    async def on_shutdown(app):
        await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
        pcs.clear()

    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    web.run_app(make_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
