# Repository Analysis Status

This document summarizes the current implementation state of this repository as observed from the checked-in files. It does not assume functionality that is not present in the repository.

## 1. Project Purpose

The repository is an undergraduate thesis project for an STM32F446RE-based wall-following / distance-estimation robot. The core research direction is embedded real-time application of Kalman filtering and adaptive Kalman filtering for fusing encoder motion input with VL53L0X time-of-flight distance measurements. The project also contains hardware verification logs, Bluetooth CSV telemetry tooling, synthetic simulation data generation, and a TinyML training pipeline for estimating adaptive measurement noise.

The top-level `README.md` currently gives only a short project description:

- STM32F446RE-based wall-following autonomous robot.
- PID control.
- VL53L0X / HC-SR04 sensor fusion.
- Undergraduate thesis context.

## 2. Folder Roles

### README

`README.md` is currently a short project title and one-line description. It does not yet document setup, firmware build, experiment execution, or analysis commands.

### simulation

`simulation/` contains Python reference models and evaluation tools.

- `kf_simulation_1D.py`: fixed 1D Kalman Filter simulation on synthetic E0 data. It generates plots and an 18-column CSV.
- `cm_akf_1D.py`: fixed KF vs covariance-matching AKF comparison on E0 synthetic data.
- `synth_data_generator.py`: synthetic E1-E5 scenario generator. It implements Fixed KF and CM-AKF, makes comparison plots, and exports CM-AKF-oriented 18-column CSV files.
- `kf_eval_metrics.py`: existing CSV evaluator. It reads a CSV with `tof_distance_mm`, `kf_estimate_mm`, `gt_distance_mm`, and `tof_residual`, then computes metrics and plots.

### firmware

`firmware/` is an STM32CubeIDE / CubeMX firmware project for STM32F446RE.

- `Core/Src/main.c`: currently a Phase 4-B motor + sensor noise measurement firmware. It initializes VL53L0X and motor PWM, then prints a 5-field test log over HC-06 / USART6.
- `Core/Src/main_loop.c`: a 200 Hz control-loop skeleton. It includes encoder reading, placeholder KF prediction/update, placeholder VL53L0X and HC-SR04 handling, motor PWM update, battery check, and DMA CSV logging.
- `Core/Src/kalman_filter.c` and `Core/Inc/kalman_filter.h`: C implementation of 1D fixed KF and CM-AKF adaptive `R`, translated from the Python simulations.
- `Drivers/VL53L0X/`: ST VL53L0X API plus STM32 I2C platform adapter.
- `Drivers/CMSIS/` and `Drivers/STM32F4xx_HAL_Driver/`: STM32 vendor support code.

### tools

`tools/` contains host-side data collection, verification, and TinyML utilities.

- `serial_logger.py`: GUI serial logger for 18-field Bluetooth CSV telemetry from the MCU.
- `verification/kf_verify.c`: PC-side C verifier comparing `kalman_filter.c` behavior against Python-generated CSV references.
- `verification/gen_verify_csv.py`: high-precision fixed KF and CM-AKF CSV generator for C verification.
- `tinyml/tinyml_train.py`: TinyML training and INT8 TFLite conversion pipeline using synthetic CSV data.
- `tinyml/tinyml_akf_3feat_int8.tflite`: checked-in INT8 TFLite model artifact.

### tests

`tests/` contains hardware verification reports, logs, images, and videos.

- `00_power_verification`: LiPo / buck converter power validation.
- `01_vl53l0x_verification`: VL53L0X I2C, initialization, repeatability, 50 Hz mode, signal rate, and range status validation.
- `02_encoder_verification`: FIT0450 encoder validation.
- `03_hcsr04_verification`: HC-SR04 trigger / echo input capture validation.
- `04A_motor_standalone_verification`: TB6612FNG motor driver standalone validation.
- `04B_motor_sensor_noise_verification`: motor-on VL53L0X noise validation.
- `05_hc06_bluetooth_verification`: HC-06 Bluetooth telemetry validation.

## 3. Currently Implemented

- Python fixed 1D KF simulation.
- Python covariance-matching CM-AKF simulation.
- Synthetic E1-E5 scenario generation.
- C fixed KF / CM-AKF implementation in `kalman_filter.c`.
- PC-side C verification scaffold for Python-to-C behavior.
- GUI serial logger that expects 18-column CSV lines from MCU telemetry.
- Hardware verification reports for power, VL53L0X, encoder, HC-SR04, motor, motor + sensor noise, and HC-06 Bluetooth.
- TinyML training script that learns `R_label` from three residual-based features and exports an INT8 TFLite model.

## 4. Partially Implemented

- `firmware/Core/Src/main_loop.c` defines a 200 Hz integrated control-loop skeleton, but it still contains placeholder sensor and KF logic.
- `main_loop.c` has DMA CSV logging, but its current output is 11 columns, not the project-wide 18-column experiment format.
- `firmware/Core/Src/main.c` is active Phase 4-B test firmware, not final experiment firmware.
- The 18-column CSV format is consistently represented in Python tools and simulations, but not yet fully emitted by the active firmware.
- TinyML-AKF exists as a training / model generation path, but not as a firmware inference path or a full offline experiment comparison path in the original files.

## 5. Not Yet Implemented

- Final MCU firmware that emits the complete 18-column experiment CSV from real sensors and filters.
- Full firmware integration of real VL53L0X non-blocking reads into the 200 Hz loop.
- Full firmware integration of `kalman_filter.c` into `main_loop.c`.
- Firmware TinyML inference runtime.
- Offline real-data 4-way comparison in the original codebase before `simulation/offline_4way_eval.py`.
- A ground-truth ingestion workflow for real experimental CSVs when `gt_distance_mm` is not generated by the MCU.

## 6. CSV Data Flow

### MCU Output

There are two relevant firmware output paths:

1. `firmware/Core/Src/main.c` currently prints Phase 4-B test data:

   ```text
   idx, vl_dist_mm, vl_status, vl_signal_MCps, sr04_us
   ```

   This is not compatible with the 18-column experiment CSV expected by `tools/serial_logger.py`.

2. `firmware/Core/Src/main_loop.c` has a DMA CSV logger with this current header:

   ```text
   tick_ms,vel_L,vel_R,kf_x,kf_v,vl53_mm,hcsr04_mm,pwm_L,pwm_R,batt_mV,loop_cyc
   ```

   This is also not the full 18-column experiment format.

### tools/serial_logger.py Storage

`tools/serial_logger.py` expects exactly 18 comma-separated fields per data line. It:

- opens a selected serial COM port at 115200 baud,
- creates a CSV file under `~/KF_Experiment_Data` by default,
- writes the 18-column header itself,
- skips an MCU header line if it starts with `timestamp_ms`,
- validates exact field count,
- validates monotonic `timestamp_ms`,
- expects approximately 20 ms spacing for 50 Hz logging,
- writes valid data lines and periodically flushes the file.

### simulation/kf_eval_metrics.py Evaluation

`simulation/kf_eval_metrics.py` reads CSV files with pandas. It requires:

```text
timestamp_ms
tof_distance_mm
kf_estimate_mm
gt_distance_mm
tof_residual
```

It computes raw sensor metrics vs ground truth and one filter estimate series vs ground truth. It can compare multiple CSV files as scenarios or repeated runs, but it does not itself rerun Fixed KF, CM-AKF, or TinyML-AKF from raw sensor input.

## 7. 18-Column CSV Format

The 18-column format used by `tools/serial_logger.py` and simulation exports is:

| # | Column | Meaning |
|---|---|---|
| 1 | `timestamp_ms` | timestamp in milliseconds |
| 2 | `tof_distance_mm` | VL53L0X distance measurement |
| 3 | `tof_signal_rate` | VL53L0X signal rate, MCPS |
| 4 | `tof_range_status` | VL53L0X range status |
| 5 | `us_distance_mm` | HC-SR04 distance |
| 6 | `encoder_distance_mm` | encoder-integrated distance |
| 7 | `encoder_speed_mms` | encoder speed input in mm/s |
| 8 | `kf_estimate_mm` | filter estimate in mm |
| 9 | `tof_residual` | residual `z - x_pred` |
| 10 | `tof_residual_var` | residual variance over window |
| 11 | `tof_residual_mean` | residual mean over window |
| 12 | `sensor_disagree` | sensor disagreement feature |
| 13 | `tof_meas_rate` | ToF measurement difference rate |
| 14 | `gt_distance_mm` | ground truth distance, when available |
| 15 | `R_label` | covariance-matching pseudo-label |
| 16 | `kalman_gain` | Kalman gain |
| 17 | `innovation_cov` | innovation covariance |
| 18 | `scenario_id` | scenario identifier |

## 8. Raw / Fixed KF / CM-AKF / TinyML-AKF Comparison Status

- Raw sensor comparison is implemented in simulation and evaluation scripts when `gt_distance_mm` exists.
- Fixed KF is implemented in Python simulation and C firmware library.
- CM-AKF is implemented in Python simulation and C firmware library.
- Synthetic Fixed KF vs CM-AKF comparison exists in `simulation/synth_data_generator.py`.
- TinyML training and INT8 conversion exists in `tools/tinyml/tinyml_train.py`.
- TinyML-AKF is not integrated into the existing firmware.
- Before `simulation/offline_4way_eval.py`, no checked-in script reran Raw, Fixed KF, CM-AKF, and TinyML-AKF together from one real experiment CSV.

## 9. Current Executable Flow for Real Experiment Data

Current files support this limited flow:

1. Use firmware that emits a valid 18-column CSV line at 50 Hz.
2. Run:

   ```bash
   python tools/serial_logger.py
   ```

3. Use the GUI to select COM port, scenario, run number, and output directory.
4. If the resulting CSV includes `gt_distance_mm`, run:

   ```bash
   python simulation/kf_eval_metrics.py path/to/input.csv
   ```

5. If multiple scenario CSVs exist:

   ```bash
   python simulation/kf_eval_metrics.py E1.csv E2.csv E3.csv E4.csv E5.csv
   ```

6. If repeated runs exist:

   ```bash
   python simulation/kf_eval_metrics.py E1_run01.csv E1_run02.csv E1_run03.csv --repeat
   ```

The new `simulation/offline_4way_eval.py` extends this by rerunning Fixed KF and CM-AKF from raw input columns and optionally attempting TinyML-AKF.

## 10. Current Biggest Bottleneck

The largest bottleneck is the gap between the intended 18-column real experiment CSV format and the currently active firmware output:

- `main.c` emits Phase 4-B 5-column diagnostic rows.
- `main_loop.c` emits an 11-column skeleton row.
- `serial_logger.py` expects exactly 18 fields.
- `kf_eval_metrics.py` requires `gt_distance_mm` for RMSE / MAE.

Until the MCU emits the full 18-column format, or a reliable post-processing conversion step exists, the analysis pipeline cannot run end-to-end on raw real experiments without manual intervention.

## 11. Thesis Chapter 4 Result-Analysis Needs

For Chapter 4 result analysis, the repository needs:

- a reproducible real-data preprocessing path from logged CSV to analysis-ready CSV,
- clear separation of Raw, Fixed KF, CM-AKF, and TinyML-AKF outputs,
- metrics per scenario and per method,
- repeated-run summaries with mean and standard deviation,
- visual plots for position estimate, residuals, adaptive `R`, and algorithm comparison,
- explicit handling for missing ground truth,
- documented limitations when ground truth is unavailable.

## 12. Next Action

Suggested next Codex tasks:

1. Add final 18-column telemetry emission to firmware without removing the existing Phase 4-B test path.
2. Add a real-data preprocessing script that can merge or assign `gt_distance_mm` for manually collected experiments.
3. Extend `simulation/offline_4way_eval.py` with a calibrated TinyML normalization/model metadata path so TinyML-AKF can run reproducibly on real CSVs.
