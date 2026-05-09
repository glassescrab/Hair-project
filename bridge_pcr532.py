import asyncio
import json
import time
from typing import Optional, Set

import serial
import websockets


SERIAL_PORT = "COM6"
BAUD_RATE = 115200
WS_HOST = "127.0.0.1"
WS_PORT = 8765
POLL_INTERVAL = 0.25
REMOVE_DELAY = 1.2

UID_TO_STORY = {
    "65B40FA7": 1,  # grey -> story 1
    "656759A7": 2,  # blue -> story 2
    "756DA5A7": 3,  # pink -> story 3
}


def hexstr(data: bytes) -> str:
    return data.hex(" ").upper()


def build_frame(cmd: int, data: bytes = b"") -> bytes:
    payload = bytes([0xD4, cmd]) + data
    length = len(payload)
    lcs = (-length) & 0xFF
    dcs = (-sum(payload)) & 0xFF
    return bytes([0x00, 0x00, 0xFF, length, lcs]) + payload + bytes([dcs, 0x00])


def read_frame(ser: serial.Serial, timeout: float = 1.0):
    deadline = time.time() + timeout

    while time.time() < deadline:
        b = ser.read(1)
        if not b or b != b"\x00":
            continue
        b2 = ser.read(1)
        if not b2 or b2 != b"\x00":
            continue
        b3 = ser.read(1)
        if not b3 or b3 != b"\xFF":
            continue

        len_byte = ser.read(1)
        lcs_byte = ser.read(1)
        if not len_byte or not lcs_byte:
            continue

        length = len_byte[0]
        lcs = lcs_byte[0]
        if length == 0x00 and lcs == 0xFF:
            post = ser.read(1)
            if post == b"\x00":
                return "ACK"
            continue
        if ((length + lcs) & 0xFF) != 0:
            continue

        payload = ser.read(length)
        dcs = ser.read(1)
        post = ser.read(1)
        if len(payload) != length or len(dcs) != 1 or post != b"\x00":
            continue
        if dcs[0] != ((-sum(payload)) & 0xFF):
            print("Bad DCS. Payload:", hexstr(payload))
            continue
        return payload

    return None


def pn532_wakeup(ser: serial.Serial):
    wake = b"\x55\x55" + b"\x00" * 14
    ser.write(wake)
    ser.flush()
    time.sleep(0.1)
    ser.reset_input_buffer()


def send_command(ser: serial.Serial, cmd: int, data: bytes = b"", timeout: float = 1.0) -> bytes:
    ser.write(build_frame(cmd, data))
    ser.flush()

    got_ack = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = read_frame(ser, timeout=max(0.01, deadline - time.time()))
        if response is None:
            break
        if response == "ACK":
            got_ack = True
            continue
        return response

    if not got_ack:
        raise TimeoutError(f"No ACK for command 0x{cmd:02X}")
    raise TimeoutError(f"No response payload for command 0x{cmd:02X}")


def get_firmware_version(ser: serial.Serial) -> Optional[bytes]:
    response = send_command(ser, 0x02, b"", timeout=1.0)
    if len(response) >= 6 and response[0] == 0xD5 and response[1] == 0x03:
        return response
    return None


def sam_normal_mode(ser: serial.Serial):
    response = send_command(ser, 0x14, b"\x01", timeout=1.0)
    if not (len(response) >= 2 and response[0] == 0xD5 and response[1] == 0x15):
        raise RuntimeError(f"SAMConfiguration failed: {hexstr(response)}")


def in_list_passive_target_type_a(ser: serial.Serial) -> Optional[str]:
    response = send_command(ser, 0x4A, b"\x01\x00", timeout=0.8)
    if not (len(response) >= 3 and response[0] == 0xD5 and response[1] == 0x4B):
        print("Unexpected InListPassiveTarget response:", hexstr(response))
        return None
    if response[2] == 0:
        return None
    if len(response) < 8:
        print("Short target response:", hexstr(response))
        return None

    nfcid_len = response[7]
    start = 8
    end = start + nfcid_len
    if len(response) < end:
        print("UID length mismatch:", hexstr(response))
        return None
    return response[start:end].hex().upper()


clients: Set[websockets.WebSocketServerProtocol] = set()


async def ws_handler(websocket):
    clients.add(websocket)
    print("Web client connected.")
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)
        print("Web client disconnected.")


async def broadcast(message: dict):
    if not clients:
        return
    dead = []
    text = json.dumps(message)
    for ws in clients:
        try:
            await ws.send(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def poll_loop():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

    try:
        pn532_wakeup(ser)
        firmware = get_firmware_version(ser)
        if firmware:
            print("Firmware response:", hexstr(firmware))
        else:
            print("Firmware response not recognized.")

        sam_normal_mode(ser)
        print("SAM normal mode set.")

        reported_uid = None
        absent_since = None

        while True:
            uid = None
            try:
                uid = in_list_passive_target_type_a(ser)
            except TimeoutError:
                pass
            except Exception as exc:
                print("Poll error:", exc)

            now = time.time()
            if uid:
                absent_since = None
                if uid != reported_uid:
                    reported_uid = uid
                    story = UID_TO_STORY.get(uid)
                    print(f"UID detected: {uid} | story={story}")
                    await broadcast({
                        "type": "nfc",
                        "present": True,
                        "uid": uid,
                        "story": story,
                        "known": story is not None,
                        "ts": now,
                    })
            elif reported_uid is not None:
                if absent_since is None:
                    absent_since = now
                elif now - absent_since >= REMOVE_DELAY:
                    print("UID removed.")
                    reported_uid = None
                    absent_since = None
                    await broadcast({
                        "type": "nfc",
                        "present": False,
                        "uid": None,
                        "story": None,
                        "known": False,
                        "ts": now,
                    })

            await asyncio.sleep(POLL_INTERVAL)
    finally:
        ser.close()


async def main():
    print(f"Starting WebSocket server on ws://{WS_HOST}:{WS_PORT}")
    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    try:
        await poll_loop()
    finally:
        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
