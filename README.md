# Camera Calibration & Perspective Measurement Assignment of CSC 8830

This is the assignment 2 for the subject -- CSc 8830 Computer Vision — Camera calibration with OpenCV,
after this this code does calibration to pull real-world 2D measurements on around 23 rectangular images.

# Camera Calibration & Measurement

Python scripts to calibrate a camera using a chessboard of size 10\*7 each and measure real-world object lengths from photos.

## Files includes

- `calibration.py`: Calibrate the camera from chessboard images.
- `measure_dimensions.py`: Measure object length from a photo using distance.
- `validate_measurements.py`: Compute error metrics (RMSE, mean error, % error).
- `app.py`: Simple Flask web UI for all steps.
- `sample_measurements.csv`: Sample CSV template for validation.
- `Module2_Report.pdf`: Assignment report and math derivations.

## Setup codes

```bash
python -m venv CVenv
source CVenv/bin/activate
pip install -r requirements.txt
```
