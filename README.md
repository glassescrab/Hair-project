# Facial Hair Exhibition Prototype

Fullscreen Python prototype for an exhibition interaction about human hair.
Idle mode shows the live camera with a blurred background and the instruction:
"Please put the dolls onto the reader to see her story". Triggered hair styles
play their matching background video, then return to the idle camera instruction
view when the video is complete.

## Setup

Install Python 3.10 or 3.11 first, then run these commands from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py
```

Terminal controls:

- `1`: short hair mode
- `2`: long hair mode
- `3`: pink straight long hair mode
- `xd`: temporary birthday message effect
- `q`: quit

NFC triggers:

- NFC polling is enabled by default on `COM7` at `115200` baud.
- The current card mapping is `65B40FA7 = 1`, `656759A7 = 2`, and `756DA5A7 = 3`.
- Terminal controls stay available while NFC polling is running.

Window controls:

- `Esc`: quit

Useful development options:

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py --windowed --duration 5
.\.venv\Scripts\python.exe hair_exhibition.py --camera 1
.\.venv\Scripts\python.exe hair_exhibition.py --backend msmf
.\.venv\Scripts\python.exe hair_exhibition.py --debug-save
.\.venv\Scripts\python.exe hair_exhibition.py --debug-save --debug-delay 1.5
.\.venv\Scripts\python.exe hair_exhibition.py --no-nfc
.\.venv\Scripts\python.exe hair_exhibition.py --nfc-port COM8
.\.venv\Scripts\python.exe hair_exhibition.py --hair-set generated
.\.venv\Scripts\python.exe hair_exhibition.py --flat-hair
.\.venv\Scripts\python.exe hair_exhibition.py --windowed --calibrate-long
```

The app saves no camera images unless `--debug-save` is enabled. Debug saving
waits for a detected face, then saves matching `_raw.png` and `_processed.png`
frames after `--debug-delay` seconds.

Hair assets:

- The app uses transparent PNGs directly from `assets/hair/`.
- Current asset mapping is `1 = short.png`, `2 = long_black.png`, and `3 = long_pink.png`.
- Direct assets keep their original colors. The app trims transparent margins and softly feathers alpha edges at runtime.
- Replace the PNG files in `assets/hair/` to update the exhibition hairstyles.

Video assets:

- Active backgrounds are read from `assets/videos/`.
- Current video mapping is `1 = short.mp4`, `2 = long.mp4`, and `3 = pink.mp4`.
- Hair modes stay active until the selected video reaches the end. If a video is missing or cannot be opened, the app falls back to `--duration`.

## Replacing Assets

To replace a hair image, export a transparent PNG and overwrite the matching
file in `assets/hair/`:

- `short.png` for mode `1`
- `long_black.png` for mode `2`
- `long_pink.png` for mode `3`

Use a PNG with an alpha channel and leave transparent space around the hair
where the face should remain visible. The app trims empty transparent margins,
scales the image from the detected face width, rotates it with the head angle,
and blends the alpha edge at runtime.

To replace a background video, overwrite the matching MP4 in `assets/videos/`:

- `short.mp4` for mode `1`
- `long.mp4` for mode `2`
- `pink.mp4` for mode `3`

Use landscape MP4 files when possible. The app resizes each video to fill the
screen and center-crops it to the camera frame, so a `16:9` video such as
`1920x1080` or `1280x720` is the easiest format to preview accurately.
