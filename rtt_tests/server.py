import argparse
import asyncio
import json
import os
import time
import math

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


# ----------------------------
# 1) 在线推理引擎：start / step
# ----------------------------
class RippleEngine:
    def __init__(self, width=640, height=360, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.speed = 1.0
        self.freq = 8.0
        self.color_shift = 0.0
        self.paused = False

        self._t0 = None
        self._frame_idx = 0

        x = np.linspace(-1.0, 1.0, self.width, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        self._rr = np.sqrt(xx * xx + yy * yy).astype(np.float32)

    def start(self):
        self._t0 = time.time()
        self._frame_idx = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def step(self):
        if self._t0 is None:
            self.start()

        if self.paused:
            t = (self._frame_idx / self.fps)
        else:
            t = (time.time() - self._t0) * float(self.speed)

        phase = self.freq * self._rr * math.pi * 2.0 - t

        r = (np.sin(phase + self.color_shift) * 0.5 + 0.5)
        g = (np.sin(phase + 2.094 + self.color_shift) * 0.5 + 0.5)
        b = (np.sin(phase + 4.188 + self.color_shift) * 0.5 + 0.5)

        frame = np.stack([r, g, b], axis=-1)
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)

        # 注意：这里 frame_idx 仍然递增；如果你希望“暂停时帧号不动”
        # 可以只在非 paused 时递增。为了直观体现暂停，这里改为：暂停就不增。
        if not self.paused:
            self._frame_idx += 1

        return frame


# ----------------------------
# 2) aiortc 视频轨：engine.step -> WebRTC
# ----------------------------
class EngineVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, engine: RippleEngine):
        super().__init__()
        self.engine = engine
        self.engine.start()

        self.fps = getattr(engine, "fps", 30)
        self._frame_interval = 1.0 / float(self.fps)
        self._last = None

    async def recv(self):
        now = time.time()
        if self._last is None:
            self._last = now
        else:
            elapsed = now - self._last
            delay = self._frame_interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.time()

        img = self.engine.step()  # RGB uint8

        # 叠字：frame_idx + paused 状态（没有 cv2 也能跑，只是不叠字）
        if cv2 is not None:
            # cv2 默认用 BGR，这里临时转一下再转回来
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            txt1 = f"frame: {self.engine._frame_idx}"
            txt2 = f"paused: {self.engine.paused}"
            cv2.putText(bgr, txt1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr, txt2, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        return frame


async def wait_ice_complete(pc: RTCPeerConnection, timeout: float = 8.0):
    if pc.iceGatheringState == "complete":
        return

    done = asyncio.Event()

    @pc.on("icegatheringstatechange")
    def _on_state_change():
        if pc.iceGatheringState == "complete":
            done.set()

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        print("[WARN] ICE gathering timeout; SDP may be missing candidates.")


pcs = set()

ICE_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        # 生产强烈建议 TURN（否则很多网络会失败）
        # RTCIceServer(
        #   urls=["turn:YOUR_TURN:3478?transport=udp"],
        #   username="USER",
        #   credential="PASS"
        # ),
    ]
)


async def index(request):
    here = os.path.dirname(os.path.abspath(__file__))
    return web.FileResponse(os.path.join(here, "client.html"))


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(ICE_CONFIG)
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_ice_state():
        print("ICE state:", pc.iceConnectionState)

    @pc.on("connectionstatechange")
    async def on_conn_state():
        print("Connection state:", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await pc.close()
            pcs.discard(pc)

    engine = RippleEngine(width=720, height=480, fps=30)
    pc.addTrack(EngineVideoTrack(engine))

    @pc.on("datachannel")
    def on_datachannel(channel):
        print("DataChannel:", channel.label)

        @channel.on("message")
        def on_message(message):
            """
            客户端发：
              {"cmd":"pause","paused":true,"id":"...","t_send_ms":12345}
            服务端回 ACK：
              {"type":"ack","id":"...","cmd":"pause","paused":true,"t_server_ms":...,"frame_idx":...}
            """
            try:
                if not isinstance(message, str):
                    return
                data = json.loads(message)

                cmd = data.get("cmd")
                msg_id = data.get("id")
                t_send_ms = data.get("t_send_ms")

                if cmd == "pause":
                    paused = bool(data.get("paused", True))
                    engine.update(paused=paused)

                    ack = {
                        "type": "ack",
                        "id": msg_id,
                        "cmd": "pause",
                        "paused": paused,
                        "t_send_ms": t_send_ms,  # 原样带回（便于客户端关联/展示）
                        "t_server_ms": int(time.time() * 1000),
                        "frame_idx": engine._frame_idx,
                    }
                    channel.send(json.dumps(ack))

                elif cmd == "set":
                    updates = {}
                    for k in ("speed", "freq", "color_shift"):
                        if k in data:
                            updates[k] = float(data[k])
                    engine.update(**updates)
                    channel.send(json.dumps({
                        "type": "ack",
                        "id": msg_id,
                        "cmd": "set",
                        "applied": updates,
                        "t_send_ms": t_send_ms,
                        "t_server_ms": int(time.time() * 1000),
                        "frame_idx": engine._frame_idx,
                    }))

                else:
                    channel.send(json.dumps({"type": "ack", "id": msg_id, "ok": False, "error": "unknown cmd"}))

            except Exception as e:
                try:
                    channel.send(json.dumps({"type": "ack", "ok": False, "error": str(e)}))
                except Exception:
                    pass

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await wait_ice_complete(pc, timeout=8.0)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
