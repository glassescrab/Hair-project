from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache" / "matplotlib"))
(PROJECT_DIR / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WINDOW_NAME = "Human Hair Exhibition"
ASSET_DIR = PROJECT_DIR / "assets" / "hair"
VIDEO_DIR = PROJECT_DIR / "assets" / "videos"
DEBUG_DIR = PROJECT_DIR / "debug_frames"
LAYERED_GENERATED_DIR = ASSET_DIR / "layered_generated"
LAYERED_DOWNLOADED_DIR = ASSET_DIR / "layered_downloaded"
BIRTHDAY_COMMAND = "xd"
BIRTHDAY_MESSAGE = "happy birthday, xd"
IDLE_MESSAGE_LINES = (
    "PICK UP A STORY.",
    "PLACE IT ON THE STAND.",
)
IDLE_STORY_CARDS = (
    ("1", "Measured", (170, 170, 170)),
    ("2", "Recognized", (102, 102, 102)),
    ("3", "Consumed", (185, 167, 244)),
)
IDLE_BASE_CACHE: dict[tuple[int, int], np.ndarray] = {}
IDLE_UI_OVERLAY_CACHE: dict[tuple[int, int], np.ndarray] = {}
FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


@dataclass(frozen=True)
class HairLayerSpec:
    file_name: str
    width_scale: float
    y_offset_ratio: float
    x_offset_ratio: float = 0.0
    restore_face_after: bool = False


@dataclass(frozen=True)
class HairMode:
    key: str
    name: str
    asset_name: str
    video_name: str
    width_scale: float
    y_offset_ratio: float
    x_offset_ratio: float = 0.0


@dataclass(frozen=True)
class FaceGeometry:
    box: tuple[int, int, int, int]
    center_x: float
    top_y: float
    width: float
    height: float
    angle_degrees: float


class VideoPlayback:
    _missing_audio_players: set[str] = set()

    def __init__(self, path: Path, play_audio: bool = True, audio_player: str = "ffplay") -> None:
        self.path = path
        self.play_audio = play_audio
        self.audio_player = audio_player
        self.capture = cv2.VideoCapture(str(path))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.started_at: float | None = None
        self.last_frame_index = -1
        self.last_fitted_frame: np.ndarray | None = None
        self.audio_process: subprocess.Popen[bytes] | None = None

    def is_opened(self) -> bool:
        return self.capture.isOpened()

    def read(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self.started_at is None:
            self.started_at = time.monotonic()
            self.start_audio()
        if self.fps > 0 and self.frame_count > 0:
            elapsed = max(0.0, time.monotonic() - self.started_at)
            target_index = int(elapsed * self.fps)
            if target_index >= self.frame_count:
                return None
            if target_index <= self.last_frame_index and self.last_fitted_frame is not None:
                return self.last_fitted_frame.copy()
            if target_index > self.last_frame_index + 1:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, target_index)

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        self.last_frame_index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.last_fitted_frame = fit_frame_to_shape(frame, shape)
        return self.last_fitted_frame.copy()

    def restart(self) -> None:
        self.stop_audio()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.started_at = None
        self.last_frame_index = -1
        self.last_fitted_frame = None

    def start_audio(self) -> None:
        if not self.play_audio or self.audio_process is not None:
            return
        player_path = shutil.which(self.audio_player)
        if player_path is None:
            if self.audio_player not in self._missing_audio_players:
                print(
                    f"Video audio disabled: '{self.audio_player}' was not found. "
                    "Install FFmpeg/ffplay or pass --audio-player with the ffplay path."
                )
                self._missing_audio_players.add(self.audio_player)
            return

        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creationflags = subprocess.CREATE_NO_WINDOW

        try:
            self.audio_process = subprocess.Popen(
                [
                    player_path,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    str(self.path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )
        except Exception as exc:
            print(f"Video audio disabled: could not start {self.audio_player}: {exc}")
            self.audio_process = None

    def stop_audio(self) -> None:
        if self.audio_process is None:
            return
        if self.audio_process.poll() is None:
            self.audio_process.terminate()
            try:
                self.audio_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.audio_process.kill()
                self.audio_process.wait(timeout=1.0)
        self.audio_process = None

    def release(self) -> None:
        self.stop_audio()
        self.capture.release()


MODES = {
    "1": HairMode(
        key="1",
        name="short",
        asset_name="wig-short.png",
        video_name="story1.mp4",
        width_scale=1.55,
        y_offset_ratio=-0.52,
    ),
    "2": HairMode(
        key="2",
        name="long",
        asset_name="wig-long-black.png",
        video_name="story2.mp4",
        width_scale=1.95,
        y_offset_ratio=-0.40,
    ),
    "3": HairMode(
        key="3",
        name="pink_long",
        asset_name="wig-long-pink.png",
        video_name="story3.mp4",
        width_scale=1.95,
        y_offset_ratio=-0.40,
    ),
}


LAYERED_SPECS = {
    "short": (
        HairLayerSpec("back.png", width_scale=1.34, y_offset_ratio=-0.53, restore_face_after=True),
        HairLayerSpec("front.png", width_scale=1.30, y_offset_ratio=-0.54, restore_face_after=True),
    ),
    "long": (
        HairLayerSpec("back.png", width_scale=1.58, y_offset_ratio=-0.50, restore_face_after=True),
        HairLayerSpec("front.png", width_scale=1.48, y_offset_ratio=-0.51),
    ),
    "pink_long": (
        HairLayerSpec("back.png", width_scale=1.58, y_offset_ratio=-0.50, restore_face_after=True),
        HairLayerSpec("front.png", width_scale=1.48, y_offset_ratio=-0.51),
    ),
}


REPLACE_HAIR_STYLES = {"long", "pink_long"}
UID_TO_MODE = {
    "65B40FA7": "1",
    "656759A7": "2",
    "756DA5A7": "3",
}
PN532_ACK_FRAME = b"\x00\x00\xFF\x00\xFF\x00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fullscreen face-tracked hair overlay prototype."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--backend",
        choices=("any", "dshow", "msmf"),
        default="dshow",
        help="OpenCV camera backend. DirectShow is the default on Windows.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Seconds to stay active for non-video effects such as the birthday message.",
    )
    parser.add_argument(
        "--debug-save",
        action="store_true",
        help="Save one processed frame per trigger for calibration.",
    )
    parser.add_argument(
        "--debug-delay",
        type=float,
        default=1.0,
        help="Seconds to wait after the first detected face before saving a debug frame.",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Use a normal window instead of fullscreen.",
    )
    parser.add_argument(
        "--flat-hair",
        action="store_true",
        help="Use the older single-PNG hair overlays instead of layered assets.",
    )
    parser.add_argument(
        "--hair-set",
        choices=("downloaded", "generated"),
        default="downloaded",
        help="Layered hair asset set to use. Ignored when --flat-hair is passed.",
    )
    parser.add_argument(
        "--calibrate-long",
        action="store_true",
        help="Capture multiple raw/processed samples for mode 2 and exit.",
    )
    parser.add_argument(
        "--calibration-count",
        type=int,
        default=6,
        help="Number of detected-face samples to capture in --calibrate-long mode.",
    )
    parser.add_argument(
        "--calibration-interval",
        type=float,
        default=1.5,
        help="Seconds between detected-face samples in --calibrate-long mode.",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=float,
        default=30.0,
        help="Maximum seconds to wait in --calibrate-long mode.",
    )
    parser.add_argument(
        "--no-nfc",
        action="store_true",
        help="Disable NFC polling and use terminal triggers only.",
    )
    parser.add_argument(
        "--nfc-port",
        default="COM6",
        help="Serial port for the PN532/PCR532 NFC reader.",
    )
    parser.add_argument(
        "--nfc-baud",
        type=int,
        default=115200,
        help="Baud rate for the PN532/PCR532 NFC reader.",
    )
    parser.add_argument(
        "--nfc-poll-interval",
        type=float,
        default=0.25,
        help="Seconds between NFC polling attempts.",
    )
    parser.add_argument(
        "--nfc-debounce",
        type=float,
        default=1.2,
        help="Seconds a removed NFC card must stay absent before returning to idle.",
    )
    parser.add_argument(
        "--no-video-audio",
        action="store_true",
        help="Disable MP4 audio playback.",
    )
    parser.add_argument(
        "--audio-player",
        default="ffplay",
        help="ffplay executable used for MP4 audio playback. Install FFmpeg or pass the full ffplay path.",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="Fullscreen output width. Defaults to the primary monitor width.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=0,
        help="Fullscreen output height. Defaults to the primary monitor height.",
    )
    return parser.parse_args()


def camera_backend(name: str) -> int:
    return {
        "any": cv2.CAP_ANY,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }[name]


def primary_display_shape() -> tuple[int, int]:
    if os.name == "nt":
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return height, width
        except Exception:
            pass
    return 720, 1280


def output_shape_from_args(args: argparse.Namespace) -> tuple[int, int]:
    if args.display_width > 0 and args.display_height > 0:
        return args.display_height, args.display_width
    if args.windowed:
        return 720, 1280
    return primary_display_shape()


def start_input_thread(command_queue: queue.Queue[str]) -> threading.Thread:
    def read_commands() -> None:
        print("Type 1, 2, 3, or xd to trigger an effect. Type q to quit.")
        while True:
            line = sys.stdin.readline()
            if line == "":
                time.sleep(0.05)
                continue
            command = line.strip().lower()
            if command:
                command_queue.put(command)

    thread = threading.Thread(target=read_commands, daemon=True)
    thread.start()
    return thread


def hexstr(data: bytes) -> str:
    return data.hex(" ").upper()


def build_pn532_frame(cmd: int, data: bytes = b"") -> bytes:
    payload = bytes([0xD4, cmd]) + data
    length = len(payload)
    lcs = (-length) & 0xFF
    dcs = (-sum(payload)) & 0xFF
    return bytes([0x00, 0x00, 0xFF, length, lcs]) + payload + bytes([dcs, 0x00])


def read_pn532_frame(ser: Any, timeout: float = 1.0) -> bytes | str | None:
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
            print("Bad PN532 DCS. Payload:", hexstr(payload))
            continue
        return payload
    return None


def pn532_wakeup(ser: Any) -> None:
    ser.write(b"\x55\x55" + b"\x00" * 14)
    ser.flush()
    time.sleep(0.1)
    ser.reset_input_buffer()


def send_pn532_command(ser: Any, cmd: int, data: bytes = b"", timeout: float = 1.0) -> bytes:
    ser.write(build_pn532_frame(cmd, data))
    ser.flush()

    got_ack = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        remaining = max(0.01, deadline - time.time())
        response = read_pn532_frame(ser, timeout=remaining)
        if response is None:
            break
        if response == "ACK":
            got_ack = True
            continue
        return response

    if not got_ack:
        raise TimeoutError(f"No ACK for PN532 command 0x{cmd:02X}")
    raise TimeoutError(f"No response payload for PN532 command 0x{cmd:02X}")


def get_pn532_firmware_version(ser: Any) -> bytes | None:
    response = send_pn532_command(ser, 0x02, b"", timeout=1.0)
    if len(response) >= 6 and response[0] == 0xD5 and response[1] == 0x03:
        return response
    return None


def set_pn532_sam_normal_mode(ser: Any) -> None:
    response = send_pn532_command(ser, 0x14, b"\x01", timeout=1.0)
    if not (len(response) >= 2 and response[0] == 0xD5 and response[1] == 0x15):
        raise RuntimeError(f"SAMConfiguration failed: {hexstr(response)}")


def read_type_a_uid(ser: Any) -> str | None:
    response = send_pn532_command(ser, 0x4A, b"\x01\x00", timeout=0.8)
    if not (len(response) >= 3 and response[0] == 0xD5 and response[1] == 0x4B):
        print("Unexpected InListPassiveTarget response:", hexstr(response))
        return None
    if response[2] == 0:
        return None
    if len(response) < 8:
        print("Short target response:", hexstr(response))
        return None

    uid_length = response[7]
    start = 8
    end = start + uid_length
    if len(response) < end:
        print("UID length mismatch:", hexstr(response))
        return None
    return response[start:end].hex().upper()


def start_nfc_thread(
    command_queue: queue.Queue[str],
    port: str,
    baud: int,
    poll_interval: float,
    debounce_seconds: float,
) -> threading.Thread:
    def poll_cards() -> None:
        try:
            import serial
        except ImportError:
            print("NFC disabled: pyserial is not installed. Run pip install -r requirements.txt.")
            return

        try:
            print(f"Opening NFC reader on {port} at {baud} baud...")
            ser = serial.Serial(port, baud, timeout=0.1)
        except Exception as exc:
            print(f"NFC disabled: could not open {port}: {exc}")
            return

        try:
            pn532_wakeup(ser)
            firmware = get_pn532_firmware_version(ser)
            if firmware:
                print("NFC firmware response:", hexstr(firmware))
            else:
                print("NFC firmware response not recognized.")
            set_pn532_sam_normal_mode(ser)
            print("NFC reader ready.")

            reported_mode: str | None = None
            absent_since: float | None = None
            while True:
                uid = None
                try:
                    uid = read_type_a_uid(ser)
                except TimeoutError:
                    pass
                except Exception as exc:
                    print("NFC poll error:", exc)

                now = time.monotonic()
                if uid:
                    absent_since = None
                    mode_key = UID_TO_MODE.get(uid)
                    if mode_key != reported_mode:
                        print(f"NFC UID detected: {uid} | mode={mode_key}")
                        command_queue.put(f"nfc:{mode_key}" if mode_key else "nfc:none")
                        reported_mode = mode_key
                elif reported_mode is not None:
                    if absent_since is None:
                        absent_since = now
                    elif now - absent_since >= max(0.0, debounce_seconds):
                        print("NFC card removed. Returning to idle.")
                        command_queue.put("nfc:none")
                        reported_mode = None
                        absent_since = None
                time.sleep(max(0.05, poll_interval))
        finally:
            ser.close()

    thread = threading.Thread(target=poll_cards, daemon=True)
    thread.start()
    return thread


def create_placeholder_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for mode in MODES.values():
        asset_path = ASSET_DIR / mode.asset_name
        if asset_path.exists():
            continue
        image = make_placeholder_hair(mode.name)
        cv2.imwrite(str(asset_path), image)
        print(f"Generated placeholder asset: {asset_path}")


def make_placeholder_hair(style: str) -> np.ndarray:
    canvas = np.zeros((520, 720, 4), dtype=np.uint8)
    hair_color = (28, 22, 18, 255)
    highlight = (72, 54, 42, 185)

    if style == "short":
        cv2.ellipse(canvas, (360, 250), (250, 185), 0, 180, 360, hair_color, -1)
        cv2.ellipse(canvas, (360, 278), (205, 128), 0, 180, 360, (0, 0, 0, 0), -1)
        for x in range(160, 570, 45):
            cv2.line(canvas, (x, 228), (x - 25, 332), highlight, 8, cv2.LINE_AA)
    elif style in {"long", "pink_long"}:
        cv2.ellipse(canvas, (360, 248), (275, 190), 0, 180, 360, hair_color, -1)
        cv2.rectangle(canvas, (105, 240), (260, 508), hair_color, -1)
        cv2.rectangle(canvas, (460, 240), (615, 508), hair_color, -1)
        cv2.ellipse(canvas, (360, 286), (210, 130), 0, 180, 360, (0, 0, 0, 0), -1)
        for x in (150, 210, 510, 570):
            cv2.line(canvas, (x, 245), (x + 18, 500), highlight, 10, cv2.LINE_AA)
    else:
        centers = [
            (190, 228),
            (250, 168),
            (320, 142),
            (395, 145),
            (468, 174),
            (530, 232),
            (176, 302),
            (540, 306),
        ]
        for center in centers:
            cv2.circle(canvas, center, 86, hair_color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (center[0] - 18, center[1] - 15), 32, highlight, 7, cv2.LINE_AA)
        cv2.ellipse(canvas, (360, 295), (205, 138), 0, 180, 360, (0, 0, 0, 0), -1)

    return canvas


def make_generated_hair_layers(style: str) -> dict[str, np.ndarray]:
    back = np.zeros((720, 900, 4), dtype=np.uint8)
    front = np.zeros((720, 900, 4), dtype=np.uint8)
    dark = (24, 18, 15, 255)
    mid = (56, 42, 33, 230)
    shine = (118, 92, 72, 120)

    if style == "short":
        cv2.ellipse(back, (450, 340), (295, 255), 0, 185, 355, dark, -1, cv2.LINE_AA)
        cv2.ellipse(back, (450, 372), (215, 160), 0, 185, 355, (0, 0, 0, 0), -1, cv2.LINE_AA)
        for x in range(210, 700, 42):
            cv2.line(back, (x, 255), (x - 34, 420), mid, 9, cv2.LINE_AA)
        cv2.ellipse(front, (450, 315), (292, 128), 0, 180, 360, dark, -1, cv2.LINE_AA)
        for x in range(220, 705, 48):
            cv2.line(front, (x, 280), (x - 40, 392), shine, 8, cv2.LINE_AA)
    elif style in {"long", "pink_long"}:
        cv2.ellipse(back, (450, 300), (330, 230), 0, 180, 360, dark, -1, cv2.LINE_AA)
        cv2.rectangle(back, (95, 305), (300, 705), dark, -1)
        cv2.rectangle(back, (600, 305), (805, 705), dark, -1)
        cv2.ellipse(back, (450, 380), (215, 155), 0, 185, 355, (0, 0, 0, 0), -1, cv2.LINE_AA)
        for x in (150, 220, 680, 750):
            cv2.line(back, (x, 315), (x + 28, 700), mid, 13, cv2.LINE_AA)
        cv2.ellipse(front, (450, 292), (284, 126), 0, 180, 360, dark, -1, cv2.LINE_AA)
        cv2.line(front, (450, 210), (420, 390), shine, 10, cv2.LINE_AA)
    else:
        for center in ((210, 290), (280, 215), (365, 178), (455, 165), (545, 188), (625, 245), (690, 322), (180, 390), (715, 410)):
            cv2.circle(back, center, 103, dark, -1, cv2.LINE_AA)
            cv2.circle(back, (center[0] - 24, center[1] - 20), 36, shine, 7, cv2.LINE_AA)
        cv2.ellipse(back, (450, 388), (220, 160), 0, 185, 355, (0, 0, 0, 0), -1, cv2.LINE_AA)
        for center in ((260, 265), (350, 215), (450, 205), (545, 225), (630, 288)):
            cv2.circle(front, center, 90, dark, -1, cv2.LINE_AA)
            cv2.circle(front, (center[0] - 20, center[1] - 18), 30, shine, 7, cv2.LINE_AA)

    return {"back.png": back, "front.png": front}


def blacken_hair_asset(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3] > 8
    if not np.any(alpha):
        return image

    luminance = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    dark_hair = np.zeros_like(image[:, :, :3])
    dark_hair[:, :, 0] = np.clip(12 + luminance * 72, 0, 92)
    dark_hair[:, :, 1] = np.clip(12 + luminance * 66, 0, 84)
    dark_hair[:, :, 2] = np.clip(14 + luminance * 58, 0, 76)
    image[:, :, :3][alpha] = dark_hair[alpha]
    return image


def pinken_hair_asset(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3] > 8
    if not np.any(alpha):
        return image

    luminance = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    contrast = np.clip((luminance - 0.18) * 1.55, 0.0, 1.0)
    pink_hair = np.zeros_like(image[:, :, :3])
    pink_hair[:, :, 0] = np.clip(42 + contrast * 96, 0, 150)
    pink_hair[:, :, 1] = np.clip(18 + contrast * 72, 0, 104)
    pink_hair[:, :, 2] = np.clip(112 + contrast * 120, 0, 238)
    image[:, :, :3][alpha] = pink_hair[alpha]
    return image


def prepare_hair_asset(image: np.ndarray, style: str) -> np.ndarray:
    if style == "pink_long":
        return feather_hair_alpha(pinken_hair_asset(image))
    return feather_hair_alpha(blacken_hair_asset(image))


def trim_hair_asset(image: np.ndarray, alpha_threshold: int = 8, padding: int = 24) -> np.ndarray:
    if image.shape[2] != 4:
        return image

    alpha = image[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0:
        return image

    x_min = max(0, int(xs.min()) - padding)
    y_min = max(0, int(ys.min()) - padding)
    x_max = min(image.shape[1], int(xs.max()) + padding + 1)
    y_max = min(image.shape[0], int(ys.max()) + padding + 1)
    trimmed = image[y_min:y_max, x_min:x_max].copy()
    trimmed[:, :, 3][trimmed[:, :, 3] <= alpha_threshold] = 0
    return trimmed


def feather_hair_alpha(image: np.ndarray) -> np.ndarray:
    image = image.copy()
    if image.shape[2] != 4:
        return image
    alpha = image[:, :, 3]
    blurred = cv2.GaussianBlur(alpha, (5, 5), 0)
    image[:, :, 3] = np.maximum(np.minimum(blurred, 245), alpha // 2).astype(np.uint8)
    return image


def load_hair_assets(
    use_layered: bool,
    hair_set: str = "downloaded",
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    assets = {}
    layered_root = LAYERED_DOWNLOADED_DIR if hair_set == "downloaded" else LAYERED_GENERATED_DIR
    for mode in MODES.values():
        path = ASSET_DIR / mode.asset_name
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.shape[2] == 3:
                alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
                image = np.concatenate([image, alpha], axis=2)
            assets[mode.key] = feather_hair_alpha(trim_hair_asset(image))
            continue

        if use_layered:
            layer_assets = {}
            mode_dir = layered_root / mode.name
            for layer in LAYERED_SPECS[mode.name]:
                image = cv2.imread(str(mode_dir / layer.file_name), cv2.IMREAD_UNCHANGED)
                if image is None:
                    layer_assets = {}
                    break
                if image.shape[2] == 3:
                    alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
                    image = np.concatenate([image, alpha], axis=2)
                layer_assets[layer.file_name] = prepare_hair_asset(image, mode.name)
            if layer_assets:
                assets[mode.key] = layer_assets
                continue

        raise RuntimeError(f"Could not load hair asset: {path}")
    return assets


def configure_window(windowed: bool) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if windowed:
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)


def enter_fullscreen() -> None:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def show_frame(frame: np.ndarray) -> int:
    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def read_camera_frame(camera: cv2.VideoCapture, fallback_shape: tuple[int, int]) -> np.ndarray:
    ok, frame = camera.read()
    if ok and frame is not None:
        return frame
    height, width = fallback_shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Camera not available",
        (60, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (120, 120, 120),
        2,
        cv2.LINE_AA,
    )
    return frame


def fit_frame_to_shape(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    source_height, source_width = frame.shape[:2]
    if source_height <= 0 or source_width <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    x0 = max(0, (resized_width - target_width) // 2)
    y0 = max(0, (resized_height - target_height) // 2)
    cropped = resized[y0 : y0 + target_height, x0 : x0 + target_width]
    if cropped.shape[:2] != (target_height, target_width):
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return cropped


def fit_mask_to_shape(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    source_height, source_width = mask.shape[:2]
    if source_height <= 0 or source_width <= 0:
        return np.zeros((target_height, target_width), dtype=np.float32)

    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(source_width * scale))
    resized_height = max(1, int(source_height * scale))
    resized = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (resized_width - target_width) // 2)
    y0 = max(0, (resized_height - target_height) // 2)
    cropped = resized[y0 : y0 + target_height, x0 : x0 + target_width]
    if cropped.shape[:2] != (target_height, target_width):
        cropped = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return cropped.astype(np.float32)


def blur_background(
    frame: np.ndarray,
    segmentation: mp.solutions.selfie_segmentation.SelfieSegmentation,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentation.process(rgb)
    mask = result.segmentation_mask
    if mask is None:
        return frame

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_3 = np.dstack([mask] * 3)
    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    return np.where(mask_3 > 0.48, frame, blurred).astype(np.uint8)


def replace_background_with_video(
    frame: np.ndarray,
    video_frame: np.ndarray,
    segmentation: mp.solutions.selfie_segmentation.SelfieSegmentation,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = segmentation.process(rgb)
    mask = result.segmentation_mask
    if mask is None:
        return video_frame.copy()

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = np.clip((mask - 0.28) / 0.52, 0.0, 1.0)
    alpha_3 = np.dstack([alpha] * 3).astype(np.float32)
    camera_float = frame.astype(np.float32)
    video_float = video_frame.astype(np.float32)
    return ((camera_float * alpha_3) + (video_float * (1.0 - alpha_3))).astype(np.uint8)


def find_face_geometry(
    frame: np.ndarray,
    face_mesh: mp.solutions.face_mesh.FaceMesh,
) -> FaceGeometry | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    height, width = frame.shape[:2]
    landmarks = result.multi_face_landmarks[0].landmark
    xs = np.array([landmark.x * width for landmark in landmarks])
    ys = np.array([landmark.y * height for landmark in landmarks])

    x_min = int(np.clip(xs.min(), 0, width - 1))
    x_max = int(np.clip(xs.max(), 0, width - 1))
    y_min = int(np.clip(ys.min(), 0, height - 1))
    y_max = int(np.clip(ys.max(), 0, height - 1))
    if x_max <= x_min or y_max <= y_min:
        return None

    def landmark_point(index: int) -> np.ndarray:
        landmark = landmarks[index]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    left_temple = landmark_point(234)
    right_temple = landmark_point(454)
    forehead = landmark_point(10)
    chin = landmark_point(152)
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    angle = np.degrees(
        np.arctan2(
            (right_eye.y - left_eye.y) * height,
            (right_eye.x - left_eye.x) * width,
        )
    )
    landmark_width = float(np.linalg.norm(right_temple - left_temple))
    landmark_height = float(np.linalg.norm(chin - forehead))
    bbox_width = float(x_max - x_min)
    bbox_height = float(y_max - y_min)
    face_width = max(landmark_width * 1.18, bbox_width * 0.82, 1.0)
    face_height = max(landmark_height * 1.08, bbox_height * 0.86, 1.0)
    center_x = float((left_temple[0] + right_temple[0]) / 2.0)
    top_y = float(min(forehead[1], y_min))

    return FaceGeometry(
        box=(x_min, y_min, x_max, y_max),
        center_x=center_x,
        top_y=top_y,
        width=face_width,
        height=face_height,
        angle_degrees=float(angle),
    )


def rotate_rgba_bound(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 0.5:
        return image

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - center[0]
    matrix[1, 2] += (new_height / 2) - center[1]
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def overlay_hair(
    frame: np.ndarray,
    hair_asset: np.ndarray,
    mode: HairMode,
    face_geometry: FaceGeometry,
) -> np.ndarray:
    face_width = max(1.0, face_geometry.width)
    face_height = max(1.0, face_geometry.height)

    overlay_width = max(24, int(face_width * mode.width_scale))
    aspect = hair_asset.shape[0] / hair_asset.shape[1]
    overlay_height = max(24, int(overlay_width * aspect))
    resized = cv2.resize(hair_asset, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)
    resized = rotate_rgba_bound(resized, -face_geometry.angle_degrees)

    x = int(face_geometry.center_x - resized.shape[1] / 2 + face_width * mode.x_offset_ratio)
    y = int(face_geometry.top_y + face_height * mode.y_offset_ratio)
    composite_rgba(frame, resized, x, y)
    return frame


def overlay_hair_layer(
    frame: np.ndarray,
    hair_asset: np.ndarray,
    layer: HairLayerSpec,
    face_geometry: FaceGeometry,
) -> None:
    face_width = max(1.0, face_geometry.width)
    face_height = max(1.0, face_geometry.height)

    overlay_width = max(24, int(face_width * layer.width_scale))
    aspect = hair_asset.shape[0] / hair_asset.shape[1]
    overlay_height = max(24, int(overlay_width * aspect))
    resized = cv2.resize(hair_asset, (overlay_width, overlay_height), interpolation=cv2.INTER_AREA)
    rotated = rotate_rgba_bound(resized, -face_geometry.angle_degrees)

    x = int(face_geometry.center_x - rotated.shape[1] / 2 + face_width * layer.x_offset_ratio)
    y = int(face_geometry.top_y + face_height * layer.y_offset_ratio)
    composite_rgba(frame, rotated, x, y)


def restore_face_oval(
    target: np.ndarray,
    source: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> None:
    x_min, y_min, x_max, y_max = face_box
    face_width = max(1, x_max - x_min)
    face_height = max(1, y_max - y_min)
    center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
    axes = (int(face_width * 0.62), int(face_height * 0.63))
    mask = np.zeros(target.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (25, 25), 0).astype(np.float32) / 255.0
    mask_3 = np.dstack([mask] * 3)
    blended = source.astype(np.float32) * mask_3 + target.astype(np.float32) * (1.0 - mask_3)
    target[:] = blended.astype(np.uint8)


def restore_visible_face(
    target: np.ndarray,
    source: np.ndarray,
    face_geometry: FaceGeometry,
) -> None:
    face_width = max(1.0, face_geometry.width)
    face_height = max(1.0, face_geometry.height)
    center = (
        int(face_geometry.center_x),
        int(face_geometry.top_y + face_height * 0.58),
    )
    axes = (int(face_width * 0.46), int(face_height * 0.48))
    mask = np.zeros(target.shape[:2], dtype=np.uint8)
    cv2.ellipse(
        mask,
        center,
        axes,
        face_geometry.angle_degrees,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    crown_cutoff = int(np.clip(face_geometry.top_y + face_height * 0.25, 0, target.shape[0] - 1))
    mask[:crown_cutoff, :] = 0
    mask = cv2.GaussianBlur(mask, (23, 23), 0).astype(np.float32) / 255.0
    mask_3 = np.dstack([mask] * 3)
    blended = source.astype(np.float32) * mask_3 + target.astype(np.float32) * (1.0 - mask_3)
    target[:] = blended.astype(np.uint8)


def composite_rgba(background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    bg_height, bg_width = background.shape[:2]
    ov_height, ov_width = overlay.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_width, x + ov_width)
    y2 = min(bg_height, y + ov_height)
    if x1 >= x2 or y1 >= y2:
        return

    ov_x1 = x1 - x
    ov_y1 = y1 - y
    ov_x2 = ov_x1 + (x2 - x1)
    ov_y2 = ov_y1 + (y2 - y1)

    overlay_crop = overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    if alpha.max() <= 0:
        return

    roi = background[y1:y2, x1:x2].astype(np.float32)
    blended = overlay_crop[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    background[y1:y2, x1:x2] = blended.astype(np.uint8)


def save_debug_frames(raw_frame: np.ndarray, processed_frame: np.ndarray, mode: HairMode, status: str) -> None:
    DEBUG_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = DEBUG_DIR / f"{stamp}_{mode.name}_{status}_raw.png"
    processed_path = DEBUG_DIR / f"{stamp}_{mode.name}_{status}_processed.png"
    cv2.imwrite(str(raw_path), raw_frame)
    cv2.imwrite(str(processed_path), processed_frame)
    print(f"Saved raw debug frame: {raw_path}")
    print(f"Saved processed debug frame: {processed_path}")


def make_contact_sheet(paths: list[Path], output_path: Path, thumb_size: tuple[int, int] = (320, 180)) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        thumb = cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA)
        label = path.stem[-32:]
        cv2.putText(thumb, label, (6, thumb_size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(thumb, label, (6, thumb_size[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return
    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.full((rows * thumb_size[1], cols * thumb_size[0], 3), 225, dtype=np.uint8)
    for index, thumb in enumerate(thumbs):
        row, col = divmod(index, cols)
        y = row * thumb_size[1]
        x = col * thumb_size[0]
        sheet[y:y + thumb_size[1], x:x + thumb_size[0]] = thumb
    cv2.imwrite(str(output_path), sheet)


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    center_y_ratio: float,
    color: tuple[int, int, int],
    outline_color: tuple[int, int, int],
    max_width_ratio: float = 0.86,
) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 5.0
    thickness = 8
    max_width = int(width * max_width_ratio)

    while scale > 0.5:
        text_width, text_height = cv2.getTextSize(text, font, scale, thickness)[0]
        if text_width <= max_width:
            break
        scale -= 0.1

    x = (width - text_width) // 2
    y = int(height * center_y_ratio + text_height / 2)
    cv2.putText(frame, text, (x, y), font, scale, outline_color, thickness + 10, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def cubic_bezier_points(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    samples: int = 36,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)[:, None]
    points = ((1 - t) ** 3 * p0) + (3 * (1 - t) ** 2 * t * p1) + (3 * (1 - t) * t**2 * p2) + (t**3 * p3)
    return points.astype(np.int32).reshape((-1, 1, 2))


def make_idle_base(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    cached = IDLE_BASE_CACHE.get((height, width))
    if cached is not None:
        return cached.copy()

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width * 0.58
    cy = height * 0.38
    radius = np.sqrt(((xx - cx) / max(1.0, width)) ** 2 + ((yy - cy) / max(1.0, height)) ** 2)
    glow = np.clip(1.0 - radius * 2.2, 0.0, 1.0)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:] = (15, 10, 10)
    purple = np.array((46, 10, 26), dtype=np.float32)
    base = base * (1.0 - glow[..., None] * 0.82) + purple * (glow[..., None] * 0.82)

    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    for index in range(22):
        x = 40 + (index / 21.0) * width
        amp = (18 + (index % 5) * 12) * width / 1280.0
        y2 = height * (0.72 + (index % 3) * 0.13)
        first = cubic_bezier_points(
            (x, 0),
            (x + amp, y2 * 0.18),
            (x - amp, y2 * 0.38),
            (x + amp * 0.6, y2 * 0.58),
        )
        second = cubic_bezier_points(
            (x + amp * 0.6, y2 * 0.58),
            (x - amp * 0.4, y2 * 0.78),
            (x + amp * 0.2, y2 * 0.9),
            (x, y2),
        )
        opacity = 0.035 + (index % 4) * 0.014
        color = tuple(int(255 * opacity) for _ in range(3))
        cv2.polylines(pattern, [first, second], False, color, max(1, int(width / 1280)), cv2.LINE_AA)

    base = cv2.add(base.astype(np.uint8), pattern)
    IDLE_BASE_CACHE[(height, width)] = base
    return base.copy()


def draw_animated_diamond(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    scale = min(width / 1280.0, height / 720.0)
    phase = time.monotonic() * (2.0 * np.pi / 2.2)
    y_offset = int(np.sin(phase) * max(5, int(9 * scale)))
    center = (width // 2, int(height * 0.047) + y_offset)
    size = max(12, int(min(width, height) * 0.022))
    color = (120, 98, 105)
    thickness = max(2, int(3 * scale))

    outer = np.array(
        [
            (center[0], center[1] - size),
            (center[0] + size, center[1]),
            (center[0], center[1] + size),
            (center[0] - size, center[1]),
        ],
        dtype=np.int32,
    )
    inner_size = int(size * 0.48)
    inner = np.array(
        [
            (center[0], center[1] - inner_size),
            (center[0] + inner_size, center[1]),
            (center[0], center[1] + inner_size),
            (center[0] - inner_size, center[1]),
        ],
        dtype=np.int32,
    )
    cv2.polylines(frame, [outer], True, color, thickness, cv2.LINE_AA)
    cv2.polylines(frame, [inner], True, color, thickness, cv2.LINE_AA)


def bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return color[2], color[1], color[0]


def load_ui_font(file_name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    key = (file_name, size)
    cached = FONT_CACHE.get(key)
    if cached is not None:
        return cached

    font_path = FONT_DIR / file_name
    if font_path.exists():
        font = ImageFont.truetype(str(font_path), size)
    else:
        font = ImageFont.load_default()
    FONT_CACHE[key] = font
    return font


def text_bbox_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def spaced_text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    spacing: int,
) -> int:
    if not text:
        return 0
    return sum(text_bbox_size(draw, char, font)[0] for char in text) + spacing * (len(text) - 1)


def draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
) -> None:
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        x += text_bbox_size(draw, char, font)[0] + spacing


def draw_text_centered_at(
    draw: ImageDraw.ImageDraw,
    width: int,
    text: str,
    center_y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
    spacing: int,
) -> None:
    text_width = spaced_text_width(draw, text, font, spacing)
    _, text_height = text_bbox_size(draw, text, font)
    x = (width - text_width) // 2
    y = center_y - text_height // 2
    draw_spaced_text(draw, (x, y), text, font, fill, spacing)


def draw_story_chips(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    scale: float,
) -> None:
    font = load_ui_font("segoeuisl.ttf", max(13, int(18 * scale)))
    gap = max(10, int(width * 0.012))
    chip_height = max(34, int(42 * scale))
    y1 = height - chip_height - max(8, int(height * 0.012))

    chips = []
    total_width = 0
    for story_id, label, color in IDLE_STORY_CARDS:
        text = f"Story {story_id} - {label}"
        text_width, _ = text_bbox_size(draw, text, font)
        chip_width = text_width + max(52, int(70 * scale))
        chips.append((text, color, chip_width))
        total_width += chip_width
    total_width += gap * (len(chips) - 1)

    x = max(8, (width - total_width) // 2)
    text_fill = (174, 170, 178)
    fill = (0, 0, 0, 112)
    for text, color_bgr, chip_width in chips:
        color = bgr_to_rgb(color_bgr)
        border = tuple(max(75, int(channel * 0.72)) for channel in color)
        x2 = x + chip_width
        y2 = y1 + chip_height
        radius = chip_height // 2
        draw.rounded_rectangle((x, y1, x2, y2), radius=radius, fill=fill, outline=border + (205,), width=max(1, int(scale)))

        dot_radius = max(4, int(5 * scale))
        dot_center = (x + int(chip_height * 0.62), y1 + chip_height // 2)
        draw.ellipse(
            (
                dot_center[0] - dot_radius,
                dot_center[1] - dot_radius,
                dot_center[0] + dot_radius,
                dot_center[1] + dot_radius,
            ),
            fill=color,
        )

        _, text_height = text_bbox_size(draw, text, font)
        text_x = x + int(chip_height * 1.02)
        text_y = y1 + (chip_height - text_height) // 2 - int(2 * scale)
        draw.text((text_x, text_y), text, font=font, fill=text_fill)
        x = x2 + gap


def make_idle_ui_overlay(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    cached = IDLE_UI_OVERLAY_CACHE.get((height, width))
    if cached is not None:
        return cached.copy()

    scale = min(width / 1280.0, height / 720.0)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    prompt_font = load_ui_font("segoeuil.ttf", max(20, int(34 * scale)))
    prompt_fill = (174, 166, 181, 255)
    letter_spacing = max(1, int(3 * scale))
    draw_text_centered_at(draw, width, IDLE_MESSAGE_LINES[0], int(height * 0.155), prompt_font, prompt_fill, letter_spacing)
    draw_text_centered_at(draw, width, IDLE_MESSAGE_LINES[1], int(height * 0.215), prompt_font, prompt_fill, letter_spacing)
    draw_story_chips(draw, width, height, scale)

    overlay = np.array(image)
    IDLE_UI_OVERLAY_CACHE[(height, width)] = overlay
    return overlay.copy()


def draw_idle_ui(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    overlay = make_idle_ui_overlay((height, width))
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = cv2.cvtColor(overlay[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    blended = overlay_bgr * alpha + frame.astype(np.float32) * (1.0 - alpha)
    result = blended.astype(np.uint8)
    draw_animated_diamond(result)
    return result


def render_idle_scene(
    frame: np.ndarray | None,
    segmentation: mp.solutions.selfie_segmentation.SelfieSegmentation | None,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if shape is not None:
        height, width = shape
    elif frame is not None:
        height, width = frame.shape[:2]
    else:
        height, width = (720, 1280)

    idle = make_idle_base((height, width))
    if frame is not None and segmentation is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = segmentation.process(rgb)
        mask = result.segmentation_mask
        if mask is not None:
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            alpha = np.clip((mask - 0.24) / 0.58, 0.0, 0.86)
            if frame.shape[:2] != (height, width):
                alpha = fit_mask_to_shape(alpha, (height, width))
                person = fit_frame_to_shape(frame, (height, width)).astype(np.float32)
            else:
                person = frame.astype(np.float32)
            alpha_3 = np.dstack([alpha] * 3)
            tint = np.full_like(person, (38, 28, 48), dtype=np.float32)
            person = person * 0.72 + tint * 0.28
            idle = (person * alpha_3 + idle.astype(np.float32) * (1.0 - alpha_3)).astype(np.uint8)

    return draw_idle_ui(idle)


def draw_birthday_effect(frame: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (20, 10, 45), -1)
    cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
    draw_centered_text(frame, "happy birthday,", 0.42, (255, 238, 120), (35, 10, 65))
    draw_centered_text(frame, "xd", 0.60, (130, 230, 255), (35, 10, 65), max_width_ratio=0.55)
    return frame


def draw_idle_message(frame: np.ndarray) -> np.ndarray:
    return draw_idle_ui(frame)


def make_idle_screen(shape: tuple[int, int]) -> np.ndarray:
    return render_idle_scene(None, None, shape)


def process_idle_frame(
    frame: np.ndarray,
    segmentation: mp.solutions.selfie_segmentation.SelfieSegmentation,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    return render_idle_scene(frame, segmentation, output_shape)


def process_active_frame(
    frame: np.ndarray,
    active_mode: HairMode,
    hair_assets: dict[str, dict[str, np.ndarray] | np.ndarray],
    face_mesh: mp.solutions.face_mesh.FaceMesh,
    segmentation: mp.solutions.selfie_segmentation.SelfieSegmentation,
    background_frame: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    face_geometry = find_face_geometry(frame, face_mesh)
    if background_frame is None:
        processed = blur_background(frame, segmentation)
    else:
        processed = replace_background_with_video(frame, background_frame, segmentation)
    if face_geometry is None:
        return processed, False

    mode_assets = hair_assets[active_mode.key]
    if isinstance(mode_assets, dict):
        base_before_hair = processed.copy()
        for layer in LAYERED_SPECS[active_mode.name]:
            overlay_hair_layer(processed, mode_assets[layer.file_name], layer, face_geometry)
            if layer.restore_face_after:
                restore_face_oval(processed, base_before_hair, face_geometry.box)
    else:
        base_before_hair = processed.copy()
        overlay_hair(processed, mode_assets, active_mode, face_geometry)
        if active_mode.name in REPLACE_HAIR_STYLES:
            restore_visible_face(processed, base_before_hair, face_geometry)
    return processed, True


def run_long_calibration(args: argparse.Namespace) -> int:
    if args.calibration_count <= 0:
        raise ValueError("--calibration-count must be greater than zero.")
    if args.calibration_interval < 0:
        raise ValueError("--calibration-interval cannot be negative.")

    create_placeholder_assets()
    hair_assets = load_hair_assets(use_layered=not args.flat_hair, hair_set=args.hair_set)
    capture_dir = DEBUG_DIR / time.strftime("long_calibration_%Y%m%d_%H%M%S")
    capture_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera, camera_backend(args.backend))
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        raise RuntimeError(f"Camera index {args.camera} could not be opened.")

    configure_window(args.windowed)
    if not args.windowed:
        enter_fullscreen()

    saved_processed: list[Path] = []
    saved_raw: list[Path] = []
    fallback_shape = (720, 1280)
    start_time = time.monotonic()
    last_capture = start_time - args.calibration_interval
    frame_index = 0

    mp_segmentation = mp.solutions.selfie_segmentation
    mp_face_mesh = mp.solutions.face_mesh
    try:
        with mp_segmentation.SelfieSegmentation(model_selection=1) as segmentation, mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        ) as face_mesh:
            while len(saved_processed) < args.calibration_count:
                now = time.monotonic()
                if now - start_time > args.calibration_timeout:
                    print("Calibration timed out before all samples were captured.")
                    break

                frame = read_camera_frame(camera, fallback_shape)
                fallback_shape = frame.shape[:2]
                processed, found_face = process_active_frame(frame.copy(), MODES["2"], hair_assets, face_mesh, segmentation)
                if found_face and now - last_capture >= args.calibration_interval:
                    frame_index += 1
                    raw_path = capture_dir / f"{frame_index:02d}_long_raw.png"
                    processed_path = capture_dir / f"{frame_index:02d}_long_processed.png"
                    cv2.imwrite(str(raw_path), frame)
                    cv2.imwrite(str(processed_path), processed)
                    saved_raw.append(raw_path)
                    saved_processed.append(processed_path)
                    last_capture = now
                    print(f"Saved calibration sample {frame_index}: {processed_path}")

                display = processed if found_face else process_idle_frame(frame, segmentation)
                cv2.putText(
                    display,
                    f"Long calibration {len(saved_processed)}/{args.calibration_count}",
                    (24, 44),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA,
                )
                key = show_frame(display)
                if key == 27:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    make_contact_sheet(saved_raw, capture_dir / "raw_sheet.png")
    make_contact_sheet(saved_processed, capture_dir / "processed_sheet.png")
    print(f"Calibration output: {capture_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")
    if args.calibrate_long:
        return run_long_calibration(args)

    fallback_shape = (720, 1280)
    output_shape = output_shape_from_args(args)
    print(f"Output canvas: {output_shape[1]}x{output_shape[0]}")
    startup_frame = make_idle_screen(output_shape)

    configure_window(args.windowed)
    show_frame(startup_frame)
    if not args.windowed:
        enter_fullscreen()
        show_frame(startup_frame)

    create_placeholder_assets()
    hair_assets = load_hair_assets(use_layered=not args.flat_hair, hair_set=args.hair_set)

    command_queue: queue.Queue[str] = queue.Queue()
    start_input_thread(command_queue)
    if not args.no_nfc:
        start_nfc_thread(
            command_queue,
            args.nfc_port,
            args.nfc_baud,
            args.nfc_poll_interval,
            args.nfc_debounce,
        )

    camera = cv2.VideoCapture(args.camera, camera_backend(args.backend))
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        print(f"Warning: camera index {args.camera} could not be opened.")

    active_effect: HairMode | str | None = None
    active_until = 0.0
    active_video: VideoPlayback | None = None
    active_from_nfc = False
    debug_saved_for_session = False
    debug_face_detected_at: float | None = None

    face_mesh = None
    mp_segmentation = mp.solutions.selfie_segmentation
    segmentation = mp_segmentation.SelfieSegmentation(model_selection=1)
    try:
        while True:
            while not command_queue.empty():
                command = command_queue.get_nowait()
                from_nfc = False
                if command.startswith("nfc:"):
                    from_nfc = True
                    command = command.removeprefix("nfc:")
                    if command == "none":
                        if active_video is not None:
                            active_video.release()
                            active_video = None
                        if active_effect is not None:
                            print("NFC idle. Returning to idle.")
                        active_effect = None
                        active_from_nfc = False
                        active_until = 0.0
                        debug_face_detected_at = None
                        continue
                if command in {"q", "quit", "exit"}:
                    return 0
                if command in MODES:
                    if from_nfc and active_from_nfc and isinstance(active_effect, HairMode) and active_effect.key == command:
                        continue
                    active_effect = MODES[command]
                    active_from_nfc = from_nfc
                    active_until = 0.0
                    if active_video is not None:
                        active_video.release()
                    video_path = VIDEO_DIR / active_effect.video_name
                    active_video = VideoPlayback(
                        video_path,
                        play_audio=not args.no_video_audio,
                        audio_player=args.audio_player,
                    )
                    if active_video.is_opened():
                        print(f"Triggered mode {command}: {active_effect.name} with video {video_path.name}")
                    else:
                        active_video.release()
                        active_video = None
                        active_until = float("inf") if from_nfc else time.monotonic() + args.duration
                        print(
                            f"Triggered mode {command}: {active_effect.name}. "
                            f"Warning: could not open {video_path}; "
                            + (
                                "staying active until NFC removal."
                                if from_nfc
                                else f"falling back to {args.duration:g} seconds."
                            )
                        )
                    debug_saved_for_session = False
                    debug_face_detected_at = None
                    if face_mesh is None:
                        mp_face_mesh = mp.solutions.face_mesh
                        face_mesh = mp_face_mesh.FaceMesh(
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.35,
                            min_tracking_confidence=0.35,
                        )
                elif command == BIRTHDAY_COMMAND:
                    if active_video is not None:
                        active_video.release()
                        active_video = None
                    active_effect = BIRTHDAY_COMMAND
                    active_from_nfc = False
                    active_until = time.monotonic() + args.duration
                    debug_saved_for_session = False
                    debug_face_detected_at = None
                    print(f"Triggered birthday effect: {BIRTHDAY_MESSAGE}")

            now = time.monotonic()
            frame = read_camera_frame(camera, fallback_shape)
            fallback_shape = frame.shape[:2]
            has_video_mode = isinstance(active_effect, HairMode) and active_video is not None
            has_timed_mode = active_effect is not None and active_video is None and now < active_until
            if has_video_mode or has_timed_mode:
                if isinstance(active_effect, HairMode):
                    background_frame = None
                    if active_video is not None:
                        background_frame = active_video.read(frame.shape[:2])
                        if background_frame is None:
                            if active_from_nfc:
                                active_video.restart()
                                background_frame = active_video.read(frame.shape[:2])
                                if background_frame is None:
                                    print("Video complete, but could not restart. Returning to idle.")
                                    active_video.release()
                                    active_video = None
                                    active_effect = None
                                    active_from_nfc = False
                                    debug_face_detected_at = None
                                    display_frame = process_idle_frame(frame, segmentation, output_shape)
                                    key = show_frame(display_frame)
                                    if key == 27:
                                        break
                                    elif key in (ord("1"), ord("2"), ord("3")):
                                        command_queue.put(chr(key))
                                    elif key == ord("x"):
                                        command_queue.put("xd")
                                    continue
                            else:
                                print("Video complete. Returning to idle.")
                                active_video.release()
                                active_video = None
                                active_effect = None
                                debug_face_detected_at = None
                                display_frame = process_idle_frame(frame, segmentation, output_shape)
                                key = show_frame(display_frame)
                                if key == 27:
                                    break
                                elif key in (ord("1"), ord("2"), ord("3")):
                                    command_queue.put(chr(key))
                                elif key == ord("x"):
                                    command_queue.put("xd")
                                continue
                    display_frame, found_face = process_active_frame(
                        frame,
                        active_effect,
                        hair_assets,
                        face_mesh,
                        segmentation,
                        background_frame,
                    )
                    if args.debug_save and found_face and debug_face_detected_at is None:
                        debug_face_detected_at = now
                    if (
                        args.debug_save
                        and found_face
                        and not debug_saved_for_session
                        and debug_face_detected_at is not None
                        and now - debug_face_detected_at >= args.debug_delay
                    ):
                        save_debug_frames(frame, display_frame, active_effect, "face")
                        debug_saved_for_session = True
                else:
                    display_frame = draw_birthday_effect(frame)
            else:
                if active_effect is not None:
                    print("Returning to idle.")
                if active_video is not None:
                    active_video.release()
                    active_video = None
                active_effect = None
                active_from_nfc = False
                debug_face_detected_at = None
                display_frame = process_idle_frame(frame, segmentation, output_shape)

            if display_frame.shape[:2] != output_shape:
                display_frame = fit_frame_to_shape(display_frame, output_shape)
            key = show_frame(display_frame)
            if key == 27:
                break
    finally:
        if active_video is not None:
            active_video.release()
        if face_mesh is not None:
            face_mesh.close()
        if segmentation is not None:
            segmentation.close()
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
