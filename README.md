# Facial Hair Exhibition Prototype

Fullscreen Python prototype for an exhibition interaction about human hair.
Idle mode shows the live camera with a blurred background and the instruction:
"Please put the dolls onto the reader to see her story". Triggered hair styles
run for 6 seconds by default, then return to the idle camera instruction view.

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

Window controls:

- `Esc`: quit

Useful development options:

```powershell
.\.venv\Scripts\python.exe hair_exhibition.py --windowed --duration 5
.\.venv\Scripts\python.exe hair_exhibition.py --camera 1
.\.venv\Scripts\python.exe hair_exhibition.py --backend msmf
.\.venv\Scripts\python.exe hair_exhibition.py --debug-save
.\.venv\Scripts\python.exe hair_exhibition.py --debug-save --debug-delay 1.5
.\.venv\Scripts\python.exe hair_exhibition.py --hair-set generated
.\.venv\Scripts\python.exe hair_exhibition.py --flat-hair
.\.venv\Scripts\python.exe hair_exhibition.py --windowed --calibrate-long
```

The app saves no camera images unless `--debug-save` is enabled. Debug saving
waits for a detected face, then saves matching `_raw.png` and `_processed.png`
frames after `--debug-delay` seconds.

Hair assets:

- The app uses downloaded layered PNGs by default from `assets/hair/layered_downloaded/`.
- Use `--hair-set generated` to compare against the generated placeholder layer set.
- Use `--flat-hair` to compare against the older single-PNG overlays.
- Hair assets are tinted at runtime so source PNG colors stay consistent. Modes `1` and `2` are black; mode `3` is pink.
- Replace `back.png` and `front.png` inside each style folder for more realistic final artwork.
