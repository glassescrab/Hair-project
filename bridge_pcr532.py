import asyncio
import json
import time
from typing import Optional, Set

import serial
import websockets

# =========================
# CONFIG
# =========================
SERIAL_PORT = "COM7"      # 改成你的 PCR532 端口
BAUD_RATE = 115200        # 不行就改成 9600 再试
WS_HOST = "127.0.0.1"
WS_PORT = 8765
POLL_INTERVAL = 0.25      # 轮询间隔（秒）

# 你的卡片映射
UID_TO_STORY = {
    "65B40FA7": 1,  # grey -> story 1
    "656759A7": 2,  # blue -> story 2
    "756DA5A7": 3,  # pink -> story 3
}

ACK_FRAME = b"\x00\x00\xFF\x00\xFF\x00"


# =========================
# PN532 FRAME HELPERS
# =========================
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
        if not b:
            continue

        if b != b"\x00":
            continue

        b2 = ser.read(1)
        if not b2 or b2 != b"\x00":
            continue

        b3 = ser.read(1)
        if not b3 or b3 != b"\xFF":
            continue

        # Could be ACK or standard frame
        len_byte = ser.read(1)
        lcs_byte = ser.read(1)
        if not len_byte or not lcs_byte:
            continue

        length = len_byte[0]
        lcs = lcs_byte[0]

        # ACK frame: 00 00 FF 00 FF 00
        if length == 0x00 and lcs == 0xFF:
            post = ser.read(1)
            if post == b"\x00":
                return "ACK"
            continue

        # Basic checksum check for LEN/LCS
        if ((length + lcs) & 0xFF) != 0:
            continue

        payload = ser.read(length)
        dcs = ser.read(1)
        post = ser.read(1)
        if len(payload) != length or len(dcs) != 1 or post != b"\x00":
            continue

        calc_dcs = (-sum(payload)) & 0xFF
        if dcs[0] != calc_dcs:
            print("Bad DCS. Payload:", hexstr(payload))
            continue

        return payload

    return None


def pn532_wakeup(ser: serial.Serial):
    # Common HSU wake-up preamble used by PN532 UART implementations
    wake = b"\x55\x55" + b"\x00" * 14
    ser.write(wake)
    ser.flush()
    time.sleep(0.1)
    ser.reset_input_buffer()


def send_command(ser: serial.Serial, cmd: int, data: bytes = b"", timeout: float = 1.0) -> bytes:
    frame = build_frame(cmd, data)
    ser.write(frame)
    ser.flush()

    got_ack = False
    deadline = time.time() + timeout

    while time.time() < deadline:
        remaining = max(0.01, deadline - time.time())
        resp = read_frame(ser, timeout=remaining)
        if resp is None:
            break
        if resp == "ACK":
            got_ack = True
            continue
        # non-ACK payload
        return resp

    if not got_ack:
        raise TimeoutError(f"No ACK for command 0x{cmd:02X}")
    raise TimeoutError(f"No response payload for command 0x{cmd:02X}")


def get_firmware_version(ser: serial.Serial) -> Optional[bytes]:
    # GetFirmwareVersion = 0x02
    resp = send_command(ser, 0x02, b"", timeout=1.0)
    # Expected: D5 03 IC Ver Rev Support
    if len(resp) >= 6 and resp[0] == 0xD5 and resp[1] == 0x03:
        return resp
    return None


def sam_normal_mode(ser: serial.Serial):
    # SAMConfiguration = 0x14 ; Normal mode = 0x01
    resp = send_command(ser, 0x14, b"\x01", timeout=1.0)
    # Expected: D5 15
    if not (len(resp) >= 2 and resp[0] == 0xD5 and resp[1] == 0x15):
        raise RuntimeError(f"SAMConfiguration failed: {hexstr(resp)}")


def in_list_passive_target_type_a(ser: serial.Serial) -> Optional[str]:
    # InListPassiveTarget = 0x4A
    # MaxTg = 1, BrTy = 0x00 (106 kbps Type A)
    resp = send_command(ser, 0x4A, b"\x01\x00", timeout=0.8)

    # Expected start: D5 4B NbTg ...
    if not (len(resp) >= 3 and resp[0] == 0xD5 and resp[1] == 0x4B):
        print("Unexpected InListPassiveTarget response:", hexstr(resp))
        return None

    nb_tg = resp[2]
    if nb_tg == 0:
        return None

    # 106 kbps Type A response structure:
    # D5 4B NbTg Tg SENS_RES(2) SEL_RES(1) NFCIDLength(1) NFCID1[...]
    if len(resp) < 8:
        print("Short target response:", hexstr(resp))
        return None

    nfcid_len = resp[7]
    start = 8
    end = start + nfcid_len
    if len(resp) < end:
        print("UID length mismatch:", hexstr(resp))
        return None

    uid = resp[start:end].hex().upper()
    return uid


# =========================
# WEBSOCKET
# =========================
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


# =========================
# MAIN POLL LOOP
# =========================
async def poll_loop():
    print(f"Opening {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

    try:
        pn532_wakeup(ser)
        fw = get_firmware_version(ser)
        if fw:
            print("Firmware response:", hexstr(fw))
        else:
            print("Firmware response not recognized.")

        sam_normal_mode(ser)
        print("SAM normal mode set.")

        last_uid = None
        last_sent_time = 0.0

        while True:
            try:
                uid = in_list_passive_target_type_a(ser)
                now = time.time()

                if uid:
                    story = UID_TO_STORY.get(uid)
                    print(f"UID detected: {uid} | story={story}")

                    # 防止同一张卡在很短时间内重复疯狂广播
                    if uid != last_uid or (now - last_sent_time) > 1.2:
                        last_uid = uid
                        last_sent_time = now
                        await broadcast({
                            "type": "nfc",
                            "uid": uid,
                            "story": story,
                            "known": story is not None,
                            "ts": now
                        })
                else:
                    # 没卡时允许下一次重新触发同一张卡
                    last_uid = None

            except TimeoutError:
                # 没刷到卡 / 一次轮询超时都很正常
                pass
            except Exception as e:
                print("Poll error:", e)

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