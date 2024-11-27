import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import pdb
import numpy as np
import cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription,VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame
import fractions

import threading
from aiohttp.web_middlewares import middleware

@middleware
async def cors_middleware(request, handler):
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all origins
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # Define allowed methods
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allow required headers
    return response
    
ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
VIDEO_CLOCK_RATE = 90000
pcs = set()
pc_states = {}
relay = MediaRelay()
VIDEO_PTIME = 1 / 20  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
def get_fake_frame():
    # 

    v = np.zeros((360, 640, 3), dtype=np.uint8)#+255
    v[:, :, 0] = 255
    return v

ACTION_LABELS = ['VERTICAL', 'HORIZONTAL', 'JUMP_OR_SNEAK', 'SPRINT', 'DROP', 'YAW', 'PITCH', 'ATTACK', 'USE', "NOISY_FRAME", "HOTBAR"]
class VideoTransformTrack(VideoStreamTrack):
    kind = "video"  # Explicitly set the kind
    preparing = True


    def __init__(self, pc_id, generate_frame_fn):
        super().__init__()  # don't forget this!
    
        self.pc_id = pc_id
        self.generate_frame_fn = generate_frame_fn
    
    async def next_timestamp(self):
        if self.readyState != "live":
            raise ValueError("Invalid state")

        if hasattr(self, "_timestamp"):
            self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, VIDEO_TIME_BASE

    async def recv(self):
        while self.pc_id not in pc_states:
            await asyncio.sleep(0.1)
        actions = pc_states.get(self.pc_id,  None) 
        # start_time = time.time()
        if self.preparing:
            img = get_fake_frame()
            # add wait a bit text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Waiting for first frame...', (10, 50), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

            frame = VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = 0
            frame.time_base = VIDEO_TIME_BASE
            self.preparing = False
            return frame

        pts, time_base = await self.next_timestamp()
        img = self.generate_frame_fn(actions) #(H, W, 3)
        
        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


import numpy as np

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    from aiortc.rtcrtpsender import RTCRtpSender
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    chosen_codec = [codec for codec in codecs if codec.mimeType == forced_codec][0]
    # chosen_codec
    transceiver.setCodecPreferences(
        [chosen_codec]
    )

def construct_action_vector(message):

    actionStates = message["actionStates"]
    viewAngles = message["viewAngles"]
    selectedInventory = message["selectedInventory"]

    FORWARD = actionStates["forward"]
    BACKWARD = actionStates["backward"]
    VERTICAL = 0 if (not FORWARD and not BACKWARD) else (1 if FORWARD else -1)

    LEFT = actionStates["left"]
    RIGHT = actionStates["right"]
    HORIZONTAL = 0 if (not LEFT and not RIGHT) else (1 if RIGHT else -1)

    JUMP = actionStates["jump"]
    SNEAK = False#actionStates["forward"]
    JUMP_OR_SNEAK = 0 if (not JUMP and not SNEAK) else (1 if JUMP else -1)

    SPRINT = int(actionStates["sprint"])
    DROP = 0.0 #int(message["actions"]["drop"])
    ATTACK = int(actionStates["attack"])
    USE = int(actionStates["use"]) # Not present in the given message
    NOISY_FRAME = 0.0  # Not present in the given message
    HOTBAR = selectedInventory#0  # Assuming this is always 0 as per the original function

    # view changes
    YAW = viewAngles["yaw"] / 5
    YAW = max(min(YAW, 3.0), -3.0)
    PITCH = viewAngles["pitch"] / 5
    PITCH = max(min(PITCH, 3.0), -3.0)

    if abs(YAW) < 0.05:
        YAW = 0
    if abs(PITCH) < 0.05:
        PITCH = 0

    # disretize the values into 0.5 segments
    bin_size = 0.2
    YAW = round(YAW / bin_size) * bin_size#(round(YAW * 2) / 2) * -1
    PITCH = round(PITCH / bin_size) * bin_size#(round(YAW * 2) / 2) * -1
    HOT_BAR_ONE_HOT = [0] * 9
    HOT_BAR_ONE_HOT[HOTBAR] = 1
    action_vector = [VERTICAL, HORIZONTAL, JUMP_OR_SNEAK, SPRINT, DROP, YAW, PITCH, ATTACK, USE, NOISY_FRAME, *HOT_BAR_ONE_HOT]

    action_vector = np.array(action_vector)

    return action_vector
last_pc_connected_at = None
end_last_peer = None
async def offer(request, diffuser_handle):
    global last_pc_connected_at, end_last_peer
    params = await request.json()
    logger.info("Offer requested, ", params)
    if len(pcs) > 0:
        return web.Response(
            status=403,
            content_type="application/json",
            text=json.dumps({"error": "Another user is currently connected. Please try again later."})
        )
    
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)
    last_pc_connected_at = time.time()
    # active_connection = pc_id

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    diffuse_step_fn, end_stream = diffuser_handle(params["past_context_noise_steps"], params["map_id"])

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            # print("message", message)
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])
            else:
                if message == "end":
                    end_stream()
                    asyncio.ensure_future(pc.close())
                    
                    pcs.discard(pc)
                else:
                    message = json.loads(message)
                    action_vector = construct_action_vector(message)
                    
                    pc_states[pc_id] =action_vector#(message["act_id"]) #action_vector

                    

    timeout_task = None

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        nonlocal timeout_task

        log_info("Connection state is %s", pc.connectionState)

        # If the state is "connecting," start a 10-second timeout
        if pc.connectionState == "connecting":
            if timeout_task is None:
                timeout_task = asyncio.create_task(connection_timeout(pc))
        
        # If connected, cancel the timeout
        elif pc.connectionState == "connected":
            if timeout_task:
                timeout_task.cancel()
                timeout_task = None

        # If failed or closed, cleanup
        elif pc.connectionState in ["failed", "closed"]:
            end_stream()
            await pc.close()
            pcs.discard(pc)

    async def connection_timeout(pc):
        await asyncio.sleep(10)
        if pc.connectionState == "connecting":
            log_info("Connection stuck in 'connecting' for over 10 seconds. Closing.")
            end_stream()
            await pc.close()
            pcs.discard(pc)

    def end_the_peer():
        end_stream()
        asyncio.ensure_future(pc.close())
        pcs.discard(pc)

    end_last_peer = end_the_peer
    

    sendable_stream = VideoTransformTrack(pc_id, diffuse_step_fn)
    sender = pc.addTrack(sendable_stream)
    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

STATIC_PATH = os.path.join(ROOT, "site")

class StrawfelServer():
    def __init__(self, diffuser_handle, port):
        self.app = web.Application()
        self.app.on_shutdown.append(on_shutdown)
        self.ssl_context = None
        self.app_port = port

        self.diffuser_handle = diffuser_handle
        self.app.router.add_route("OPTIONS", "/offer", self.options_handler)
        self.app.router.add_post("/offer", self.wrapped_offer)
        self.app.middlewares.append(cors_middleware)
        self.app.router.add_get("/", self.serve_index)
        self.app.router.add_static("/", STATIC_PATH, show_index=False)


    async def serve_index(self, request):
        """Serve the index.html file for the root path."""
        index_path = os.path.join(STATIC_PATH, "index.html")
        return web.FileResponse(index_path)


    async def wrapped_offer(self, request):
            return await offer(request, self.diffuser_handle)  # Use the diffused frame here

    async def options_handler(self, request):
        headers = {
            "Allow": "OPTIONS, POST",
            "Access-Control-Allow-Origin": "*",   # Adjust CORS policy as needed
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
        return web.Response(status=204, headers=headers)
    async def start(self):
        test, end = self.diffuser_handle(0.15, "759_384_0")
        ACTION_LABELS = ['VERTICAL', 'HORIZONTAL', 'JUMP_OR_SNEAK', 'SPRINT', 'DROP', 'YAW', 'PITCH', 'ATTACK', 'USE', "NOISY_FRAME", *([f"HOTBAR{i}" for i in range(9)])]
        for i in range(10):
            test(np.zeros(len(ACTION_LABELS)))
        end()
        print("warmed up")

        logger.log(logging.INFO, "Starting server")
        runner = web.AppRunner(self.app)
        await runner.setup()
        logger.log(logging.INFO, f"Server started at http://127.0.0.1:{self.app_port}")
        site = web.TCPSite(runner, '127.0.0.1', self.app_port)

        await site.start()
        await asyncio.Event().wait()

    def connection_mgr(self):
        return end_last_peer, last_pc_connected_at, len(pcs) > 0



    async def peer_work(self, clbk):
        print("peer process started")
        while True:
            await asyncio.sleep(0.1)
            clbk()
    def step(self, clbk):
        clbk()
        if len(pcs) > 0:
            asyncio.run(self.peer_work(clbk))
        else:
            return

def start_background_loop(loop: asyncio.AbstractEventLoop, strwfl: StrawfelServer) -> None:
    asyncio.set_event_loop(loop)
    
    asyncio.run(strwfl.start())
    loop.run_forever()

import time

def start_demo_thingy(dhandle, port):
    logging.basicConfig(level=logging.INFO)

    strwfl = StrawfelServer(dhandle, port)
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_background_loop, args=(loop,strwfl), daemon=True)
    t.start()
    return strwfl
