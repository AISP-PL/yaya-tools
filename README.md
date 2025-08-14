# YAYA tools - package with set of AISP yet another YOLO annotations tools


# Installation : User

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install YAYA tools directly from github releases.
For example

```bash
pip install https://github.com/AISP-PL/yaya-tools/releases/download/v1.0.4/yaya_tools-0.1.0-py3-none-any.whl
```

# Installation : Developer

Download repository and install via poetry packages with --dev group.
For example

```bash
git clone git@github.com:AISP-PL/yaya-tools.git
cd yaya-tools
poetry install
```

# Run
Use the following commands after installing the package:

## YAYA test

```bash
# Test package installation
yaya-test

2025-08-14 17:19:31,632 INFO: 
###### Logging start of terminal session ######
yaya_tools package installed successfully!
```

## YAYA readme

```bash
usage: yaya-readme [-h] [-v] [--update KEY VALUE] [-f FILE]

yaya-readme: Command‐line tool to update or append a key/value in a text file (e.g. readme.kmd).

options:
  -h, --help            Show this help message and exit.
  -v, --version         Show program's version number and exit.
  --update KEY VALUE    Update the given KEY with VALUE. If KEY exists, replaces its entire line. Otherwise, appends at the end.
  -f FILE, --file FILE  Path to the file to update (default: README.md).
```

## YAYA dataset management

```bash
usage: yaya-dataset -i DATASET_PATH [--fix_toosmall] [--fix_xywh_normalization] [--fix_xyxy_normalization] [--copy_negatives_to COPY_NEGATIVES_TO] [--train_all] [--validation_force_create] [--ratio RATIO] [-h] [-v]

YAYa dataset management tool

options:
  -i DATASET_PATH, --dataset_path DATASET_PATH
                        Path to the dataset folder
  --fix_toosmall        Fix too small annotations
  --fix_xywh_normalization
                        Fix xywh normalization
  --fix_xyxy_normalization
                        Fix xyxy normalization
  --copy_negatives_to COPY_NEGATIVES_TO
                        Path to copy the negative samples only
  --train_all           Use all images for training dataset
  --validation_force_create
                        Recreate the validation.txt file from the dataset folder
  --ratio RATIO         Validation ratio (default=0.2)
  -h, --help            Show this help message and exit.
  -v                    Show version and exit.
```

## YAYa dataset diff

```bash
usage: yaya-datasetdiff -s SOURCE -d DEST [--add_new] [--remove_old] [-h] [-v]

YAYa dataset management tool

options:
  -s SOURCE, --source SOURCE
                        Path to the source dataset folder
  -d DEST, --dest DEST  Path to the destination dataset folder
  --add_new             Add only new annotations to the destination
  --remove_old          Remove only old annotations from the destination
  -h, --help            Show this help message and exit.
  -v                    Show version and exit.
```

## YAYA resize

```bash
usage: yaya-resize -i DATASET_PATH -o OUTPUT_PATH [--width WIDTH] [--height HEIGHT] [--copy_annotations] [-h] [-v]

YAYa dataset management tool

options:
  -i DATASET_PATH, --dataset_path DATASET_PATH
                        Path to the dataset folder
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the output folder
  --width WIDTH         Width of the resized image
  --height HEIGHT       Height of the resized image
  --copy_annotations    Copy annotations if found to the output folder
  -h, --help            Show this help message and exit.
  -v                    Show version and exit.
```

## YAYA benchmark

```bash
usage: yaya-benchmark -d DATASET --cfg_path CFG_PATH --weights_path WEIGHTS_PATH --names_path NAMES_PATH [--confidence CONFIDENCE] [--nms_threshold NMS_THRESHOLD] [-h] [-v]

YAYa dataset management tool

options:
  -d DATASET, --dataset DATASET
                        Path to the dataset folder
  --cfg_path CFG_PATH   Path to the configuration file
  --weights_path WEIGHTS_PATH
                        Path to the weights file
  --names_path NAMES_PATH
                        Path to the names file
  --confidence CONFIDENCE
                        Confidence threshold
  --nms_threshold NMS_THRESHOLD
                        NMS threshold
  -h, --help            Show this help message and exit.
  -v                    Show version and exit.
```


## YAYA inference

# Run inference on video
yaya-inference --video <INPUT_VIDEO> --output <OUTPUT_VIDEO> --cfg_path <CFG> --weights_path <WEIGHTS> --names_path <NAMES> [--confidence <CONF>] [--nms_threshold <NMS>] [--tracking] [--gpu]

## YAYA augment

# Augment dataset (selection + augmentation flags)
yaya-augument -i <DATASET_PATH> [selection flags] [augmentation flags] [-n <ITERATIONS>]

# GUI inference (Qt5)
yaya-inference-qt5

# GUI Darknet log comparison (Qt5)
yaya-darknet-logs-qt5 --log1 <LOG1_PATH> --log2 <LOG2_PATH>
```


