# Facial Hair Exhibition Prototype

Fullscreen Python prototype for an exhibition interaction about human hair.
Idle mode shows the live camera with a blurred background and the instruction:
"Please put the dolls onto the reader to see her story". Triggered hair styles
play their matching background video with audio. NFC interaction stays in the
selected mode while the doll remains on the reader and returns to the idle
camera instruction view when the doll is removed.

## Setup

Install Python 3.10 or 3.11 first. Do not use Python 3.14 for this project:
`mediapipe==0.10.14` needs a Python 3.11-compatible wheel.

On Windows, this installs the tested runtime:

```powershell
winget install --id Python.Python.3.11 --source winget -e --accept-package-agreements --accept-source-agreements
```

Install FFmpeg too. The app uses `ffplay` from FFmpeg to play the audio track
from the MP4 background videos while OpenCV renders the video frames:

```powershell
winget install --id Gyan.FFmpeg --source winget -e --accept-package-agreements --accept-source-agreements
```

Open a new PowerShell after installing Python or FFmpeg so PATH changes are
loaded. Then run these commands from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check that the environment is ready:

```powershell
.\.venv\Scripts\python.exe -m pip check
ffplay -version
```

If `ffplay -version` is not found after installing FFmpeg, use the full
`ffplay.exe` path when starting the app. The default `winget` install path is
usually:

```text
C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffplay.exe
```

Use this command if `ffplay` is not on `PATH`:

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py --audio-player "C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffplay.exe"
```

## Run

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py
```

If MP4 audio does not play with the normal run command, run with the full
`ffplay.exe` path:

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py --audio-player "C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffplay.exe"
```

Terminal controls:

- `1`: short hair mode
- `2`: long hair mode
- `3`: pink straight long hair mode
- `xd`: temporary birthday message effect
- `q`: quit

NFC triggers:

- NFC polling is enabled by default on `COM6` at `115200` baud.
- The current card mapping is `65B40FA7 = 1`, `656759A7 = 2`, and `756DA5A7 = 3`.
- The current mode follows reader state: card present selects its mapped mode,
  and card removed returns to idle. The same card staying on the reader does not
  retrigger the video.
- Terminal controls stay available while NFC polling is running.

Window controls:

- `Esc`: quit

Display resolution:

- Fullscreen mode renders to the primary monitor resolution so the idle prompt,
  icon, and story chips stay sharp instead of being stretched from the camera
  frame. If monitor detection is wrong, override it with `--display-width` and
  `--display-height`.

Useful development options:

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py --windowed --duration 5
.\.venv\Scripts\python.exe hair_exhibition.py --camera 1
.\.venv\Scripts\python.exe hair_exhibition.py --backend msmf
.\.venv\Scripts\python.exe hair_exhibition.py --debug-save
.\.venv\Scripts\python.exe hair_exhibition.py --debug-save --debug-delay 1.5
.\.venv\Scripts\python.exe hair_exhibition.py --no-nfc
.\.venv\Scripts\python.exe hair_exhibition.py --nfc-port COM8
.\.venv\Scripts\python.exe hair_exhibition.py --audio-player C:\ffmpeg\bin\ffplay.exe
.\.venv\Scripts\python.exe hair_exhibition.py --no-video-audio
.\.venv\Scripts\python.exe hair_exhibition.py --display-width 1920 --display-height 1080
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
- Current video mapping is `1 = story1.mp4`, `2 = story2.mp4`, and `3 = story3.mp4`.
- MP4 audio is played through `ffplay`, so FFmpeg must be installed and
  `ffplay` must be on `PATH`, or pass its full path with `--audio-player`.
- While a story video is playing, the camera/person hair effect is shown only
  during the configured opening and ending windows. Edit
  `VIDEO_CAMERA_OVERLAY_START_SECONDS` and `VIDEO_CAMERA_OVERLAY_END_SECONDS`
  near the top of `hair_exhibition.py` to set those lengths independently. The
  middle of the video plays unobstructed. Videos shorter than the sum of the two
  windows keep the camera/person hair effect visible for the full video because
  the two windows overlap.
- NFC hair modes stay active until the card is removed. When the video reaches
  the end while the card is still present, it loops without an NFC retrigger. If
  a video is missing or cannot be opened, NFC mode keeps the hair effect active
  until removal; terminal-triggered modes fall back to `--duration`.

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

- `story1.mp4` for mode `1`
- `story2.mp4` for mode `2`
- `story3.mp4` for mode `3`

Use landscape MP4 files when possible. The app resizes each video to fill the
screen and center-crops it to the camera frame, so a `16:9` video such as
`1920x1080` or `1280x720` is the easiest format to preview accurately.
