# Computer Vision Homeworks (CSc 8830)

A single Flask app that hosts multiple homeworks, each as its own module/blueprint, all reachable from one
landing page.

## Structure

```
app.py                          # app factory, registers one blueprint per homework
homeworks/
  hw2_calibration/               # Homework 2 - camera calibration & perspective measurement
    calibration.py               # chessboard corner detection + cv2.calibrateCamera
    measure_dimensions.py        # undistort points, pixel->real-world length
    validate_measurements.py     # error stats + plots vs. ground truth
    routes.py                    # Flask blueprint (/hw2/...)
  hw3_blurring/                  # Homework 3 - blurring: spatial filter vs. Fourier domain
    blur.py                      # spatial convolution, FFT-domain convolution, spectra
    routes.py                    # Flask blueprint (/hw3/...)
  hw4_segmentation/               # Homework 4 - classical human boundary segmentation vs. SAM2
    rgb_segmentation.py          # GrabCut-based segmentation for color photos
    thermal_segmentation.py      # Otsu-threshold segmentation for thermal frames
    metrics.py                   # IoU / Dice / overlay against an externally-generated SAM2 mask
    routes.py                    # Flask blueprint (/hw4/...)
templates/
  base.html, landing.html         # shared shell + homework index
  hw2/, hw3/, hw4/                 # per-homework pages, each with its own sidebar layout
static/
  css/style.css                   # shared stylesheet
  calib/, calib_uploads/, object_uploads/, results/   # hw2 runtime data
  hw3/uploads/, hw3/outputs/       # hw3 runtime data
  hw4/uploads/, hw4/outputs/       # hw4 runtime data
```

Each homework is self-contained under `homeworks/<name>/` with its own routes and templates, and is registered
as a Flask blueprint in `app.py`.

## Homework 2 - Camera Calibration & Measurement

Calibrates a camera from chessboard photos, then uses the intrinsics plus a known camera-to-object distance to
turn a pixel measurement into a real-world length. Validated against a 20+ object ground-truth CSV.

## Homework 3 - Image Blurring: Spatial Filter vs. Fourier Domain

Blurs an uploaded image two independent ways on the same kernel:
- **Spatial**: direct 2D convolution (`cv2.filter2D`).
- **Frequency**: zero-pad the image and kernel, multiply their FFTs, inverse-FFT, crop back (the standard
  "linear convolution via padded circular convolution" trick, needed because a bare FFT multiply gives circular,
  not linear, convolution).

The page reports the pixel-wise difference between the two results (typically ~1e-12, i.e. floating-point noise)
as the experimental confirmation of the convolution theorem `f * h <-> F . H`, alongside magnitude-spectrum
visualizations. The `/hw3/theory` page has the full derivation.

## Homework 4 - Human Boundary Segmentation (Classical CV) vs. SAM2

Two classical (non-ML, non-DL) OpenCV segmentation pipelines, each with a boundary/contour output:
- **RGB**: GrabCut (iterative graph-cut / GMM energy minimization on a single image, no trained model) seeded
  with a rectangle you drag on the photo, or an automatic whole-image box.
- **Thermal**: Otsu global thresholding + morphology - a person is almost always the warmest (brightest) blob in
  a thermal frame.

Both pages accept an optional mask generated separately via
[Meta's SAM2](https://ai.meta.com/research/sam2/) (e.g. the [SAM2 web demo](https://sam2.metademolab.com/): upload
the same photo, click the person, export the mask) and compute IoU/Dice plus a visual agreement overlay against
it. SAM2 is a deep model, so it's used only as an external comparison reference, never inside our own
implementation. `/hw4/theory` derives edge detection and region segmentation from Fourier-domain filtering.

## Setup

```bash
python -m venv CVenv
source CVenv/bin/activate
pip install -r requirements.txt
python app.py    # http://localhost:5000
```
