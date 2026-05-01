# Strategem Master

Strategem Master is a Windows desktop overlay for Helldivers 2. It captures the
game HUD, detects equipped stratagem icons, shows them in a transparent overlay,
and can send the selected stratagem key sequence through hotkeys.

The runtime detector uses icon template matching. OCR is used only by the asset
pipeline to extract and name icon templates from source screenshots.

## Features

- Transparent PyQt5 overlay for detected stratagem slots.
- Dynamic HUD crop that adapts better to different screen sizes.
- Icon detection with OpenCV, Canny/Sobel edge features, CLAHE, auto Canny, and
  optional pHash shortlist in the accuracy test tool.
- OCR-only asset extraction pipeline for generating `img/<code>.png` templates.
- No OCR manifest, override, or alias mapping in the extraction flow.
- Hotkeys for rescanning, slot activation, default stratagems, and debug boxes.
- Accuracy test runner for comparing old and new detection against samples.

## Project Layout

```text
main.py                         Runtime overlay app
overlay_window.py               Overlay UI
icon_regions_overlay.py         Debug overlay for detected icon boxes
img/                            Runtime icon templates named by stratagem code
src/strategems.csv              Only CSV source for stratagem metadata
src/img/group/                  Source screenshots used by the asset pipeline
src/utils/automation_pipeline.py OCR-only asset update/extraction pipeline
src/utils/strategem_detection.py Shared old/new detection logic for tests
src/utils/screen_regions.py     Dynamic HUD crop helpers
tools/auto_update_manager.py    Auto-update checker/manager
.test/accuracy_test.py          Accuracy comparison runner
.test/sample/                   Test images and samples.json
output/pipeline.log             Runtime pipeline log
```

## Data And Assets

The project uses one CSV file only:

```text
src/strategems.csv
```

Each row must contain:

```csv
Index,OriginalName,Name,Code
20,MGX-42 Bullet Storm,Bullet Storm,414321
```

`OriginalName` is the full wiki name. `Name` is generated after the project
rule and is used by the runtime UI, logs, and detection results.

Runtime icon templates are stored in:

```text
img/<Code>.png
```

Example:

```text
img/414321.png
```

If `src/strategems.csv` or `img/*.png` is missing, `main.py` will stop and ask
you to run the asset pipeline.

## Installation

1. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Install Tesseract OCR for the asset pipeline.

Default path expected by the pipeline:

```text
C:\My Programs\Tesseract-OCR\tesseract.exe
```

If your Tesseract path is different, update `TESSERACT_PATH` in:

```text
src/utils/automation_pipeline.py
```

## Generate Or Update Assets

Put source screenshots in:

```text
src/img/group/
```

Then run:

```powershell
.\.venv\Scripts\python.exe src\utils\automation_pipeline.py
```

The pipeline will:

- Fetch/update `src/strategems.csv`.
- Generate UI names from wiki names, such as `AX/ARC-3 K-9` -> `K-9`.
- OCR stratagem names from screenshots in `src/img/group/`.
- Save extracted icons into `img/` as `<code>.png`.
- Skip icon files that already exist.
- Write logs to `output/pipeline.log`.

The OCR extractor is intentionally OCR-only. It does not use manifest files,
aliases, or manual code overrides.

To verify local assets and only run the pipeline when an update is needed:

```powershell
.\.venv\Scripts\python.exe tools\auto_update_manager.py
```

## Run The Overlay

```powershell
.\.venv\Scripts\python.exe main.py
```

Run the terminal as administrator if global hotkeys or simulated key input do
not work in game.

## Hotkeys

| Hotkey | Action |
| --- | --- |
| `Ctrl+]` | Capture the Helldivers 2 window and update detected stratagem slots |
| `Ctrl+[` | Show debug overlay for detected icon boxes |
| `Ctrl+1` to `Ctrl+0` | Activate detected stratagem by slot |
| `Ctrl+g` | Activate Reinforce (`24312`) |
| `Ctrl+v` | Activate Resupply (`4423`) |
| `Ctrl+q` | Activate Eagle Rearm (`22123`) |
| `Ctrl+c` | Exit |

Default stratagems such as Reinforce, Resupply, SoS Beacon, and Eagle Rearm are
filtered out of normal detected slots.

## Accuracy Test

Keep test samples in:

```text
.test/sample/
```

Expected codes are defined in:

```text
.test/sample/samples.json
```

Run the default accuracy comparison:

```powershell
.\.venv\Scripts\python.exe .test\accuracy_test.py
```

Verbose output:

```powershell
.\.venv\Scripts\python.exe .test\accuracy_test.py --verbose
```

Include default stratagems in the comparison:

```powershell
.\.venv\Scripts\python.exe .test\accuracy_test.py --include-defaults --skip-first 0
```

Save a JSON report:

```powershell
.\.venv\Scripts\python.exe .test\accuracy_test.py --report output\accuracy_report.json
```

Save debug overlay images:

```powershell
.\.venv\Scripts\python.exe .test\accuracy_test.py --debug-dir output\accuracy_debug --verbose
```

## OCR Notes

The asset pipeline improves OCR by generating multiple text variants:

- CLAHE for local contrast balancing.
- Bilateral filtering to reduce noise while preserving text edges.
- Dynamic white-text masks for white HUD text on translucent black backgrounds.
- Otsu and adaptive threshold variants.
- Multiple Tesseract page segmentation modes.
- Fuzzy matching against names from `src/strategems.csv`.

This helps with glare and uneven illumination, but OCR can still fail if the
text is heavily washed out or cropped. The current flow still stays OCR-only and
does not use manual aliases or manifest overrides.

## Troubleshooting

- `File not found: ./src/strategems.csv`: run the asset pipeline or restore the
  CSV at `src/strategems.csv`.
- `No icon images (.png) found in ./img/`: run the asset pipeline or add
  templates named by code, for example `img/414321.png`.
- Hotkeys do nothing: run PowerShell or the app as administrator.
- Helldivers window is not found: make sure the game is running and visible.
- OCR extraction misses an icon: add a clearer source screenshot to
  `src/img/group/` and rerun the pipeline.
- Low accuracy at small/windowed resolutions: add samples to `.test/sample/`,
  update `samples.json`, and run the accuracy test to compare changes.

## License

MIT License. See `LICENSE.txt`.
