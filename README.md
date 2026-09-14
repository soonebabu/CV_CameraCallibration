# Camera Calibration & Perspective Measurement

CSc 8830 Computer Vision — Module 2 assignment. Camera calibration with OpenCV,
then using that calibration to pull real-world 2D measurements out of a plain
photo, and finally validating how accurate that turns out to be over 20
objects.

Everything below can be run either from the command line or from the small
Flask web app included in this repo (`app.py`) — the app is just a thin UI
wrapper around the same three scripts.

## What's in here

| File | Purpose |
|---|---|
| `calibration.py` | Step 1 — OpenCV chessboard calibration from a folder of phone photos |
| `measure_dimensions.py` | Step 2 — turns a pixel measurement + known distance into a real-world length |
| `validate_measurements.py` | Step 3 — error statistics (mean/RMSE/% error) over the 20-object validation set |
| `app.py` + `templates/` | Web app tying the three steps together in a browser |
| `sample_measurements.csv` | Template for the validation CSV — fill in your own numbers |
| `Module2_Report.pdf` | Write-up: methodology, results, and the stereo-camera theory derivation |

Each script also has a docstring at the top explaining exactly what it does,
what arguments it takes, and how to run it — that's the "ReadMe on top of
each script" the assignment asks for, so it travels with the code instead of
living only here.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Step 1 — Calibration

Print a chessboard (this repo assumes the common 9x6-inner-corner board —
change `--cols`/`--rows` if yours is different), tape it to something flat,
and take 15–25 photos with your phone from different angles and distances.
Auto-focus is fine, just don't change zoom level between shots. Drop the
photos in `calibration_images/` and run:

```bash
python calibration.py --images calibration_images --cols 9 --rows 6 --square 25.0
```

This prints the camera matrix (fx, fy, cx, cy), the distortion coefficients,
and the mean reprojection error, and saves everything to `camera_calib.npz`
for the next step to use.

Aim for a reprojection error under ~1 pixel. If it's worse, the usual
culprits are motion blur on a few shots or a wrong `--cols`/`--rows` value.

## Step 2 — Measuring an object

```bash
python measure_dimensions.py --image object_images/box1.jpg --distance 2500 --calib camera_calib.npz
```

`--distance` is how far the camera was from the object plane, in mm,
measured with a tape measure or laser distance meter at capture time. A
window pops up — click the two endpoints of the edge you want measured and
press any key. The script undistorts those points using the calibration
result and applies the pinhole projection formula:

```
L = (pixel_distance x Z) / f
```

where `f` is the calibrated focal length in pixels (fx or fy) and `Z` is the
distance you supplied. The full derivation and the assumptions behind it
(fronto-parallel object plane, distortion-corrected pixels) are in the
report.

## Step 3 — Validating accuracy

Measure 20 different objects at a known distance (anything past 2 m per the
assignment), record the real length with a tape measure, and record what
`measure_dimensions.py` computed for each. Fill those into a CSV shaped like
`sample_measurements.csv`:

```
object_id,actual_mm,measured_mm,distance_mm
book_width,180,176.4,2500
...
```

then run:

```bash
python validate_measurements.py --csv my_measurements.csv --out results
```

This writes `results/validation_summary.txt` (mean error, RMSE, % error,
std. dev.), `results/validation_detail.csv`, and two plots comparing actual
vs. measured length and per-object percent error.

## Running the web app

```bash
python app.py
```

Then open `http://127.0.0.1:5000`. The nav bar has a page for each step:
upload chessboard photos and calibrate, upload an object photo and click two
points on it to measure, or upload a validation CSV and see the error
report. State is kept as a single shared calibration file under
`static/calib/` — this is a one-user local demo, not meant to be
multi-tenant.

## A few things worth knowing before you run this for real

- The pinhole formula in Step 2 assumes the measured edge is roughly
  perpendicular to the camera's optical axis. Hold the phone square-on to
  the object, not at a steep angle, or the length will be off.
- Keep the same zoom level / focal length between calibration and the
  actual measurement shots — the calibration is only valid for the optical
  setup it was computed from.
- If your phone does any digital "AI" zoom or HDR multi-frame stacking, turn
  it off if you can; those can shift the effective focal length between
  shots in ways that aren't captured by a static camera matrix.
