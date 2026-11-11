# Video Keyframe Detection and Extraction

A small Python script that selects video frames for Stable Diffusion / EbSynth workflows using the Structural Similarity Index (SSIM). It saves the original-resolution frames in `synth_frames`.

The **first frame is always saved**. Each later sampled frame is compared with the **last saved keyframe**, not merely the preceding video frame. This catches gradual changes that accumulate over time without requiring a sudden scene cut.

## Installation

Requires Python 3:

```bash
pip install opencv-python-headless scikit-image numpy tqdm
```

## Usage

Edit the settings at the top of `ssim_keyframe_extraction.py`:

```python
INPUT_VIDEO = 'path/to/input/video.mp4'
FRAME_RATE = 1
THRESHOLD = 0.90
OUTPUT_PATH = 'path/to/output/folder'
```

- `INPUT_VIDEO`: source video path.
- `OUTPUT_PATH`: parent directory for the `synth_frames` folder.
- `FRAME_RATE`: approximate number of video frames **per second to analyse**, not a playback delay. Default: one per second. Values at or above the source FPS analyse every frame.
- `THRESHOLD`: SSIM similarity cutoff from `-1` to `1`. Save a sampled frame when its similarity to the last saved keyframe is **below** this value. A higher threshold generally saves **more** frames. The default `0.90` is a starting point.

Run:

```bash
python ssim_keyframe_extraction.py
```

Frames are saved as PNGs at original resolution. Filenames contain the zero-based source video frame number, padded for sorting. Comparisons use downscaled colour frames for speed.

**Before rerunning:** start with an empty output directory (or clear the old `synth_frames` PNGs). This script deliberately does not delete older outputs automatically.

The script selects frames only. Stable Diffusion and EbSynth remain separate tools.

## Tests

With `pytest` installed, run:

```bash
python -m pytest -q
```
