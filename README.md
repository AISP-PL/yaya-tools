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

```bash
# Test package installation
yaya-test

# Update or append a key/value in a text file
yaya-readme --update <KEY> <VALUE> [-f <FILE>]

# Manage dataset (summary, fixes, train/validation lists)
yaya-dataset -i <DATASET_PATH> [--fix_toosmall] [--fix_xywh_normalization] [--fix_xyxy_normalization] [--copy_negatives_to <PATH>] [--train_all] [--validation_force_create] [--ratio <RATIO>]

# Diff datasets: add new/remove old annotations
yaya-datasetdiff -s <SOURCE> -d <DEST> [--add_new] [--remove_old]

# Resize annotated images
yaya-resize -i <DATASET_PATH> -o <OUTPUT_PATH> [--width <W>] [--height <H>] [--copy_annotations]

# Benchmark detector on dataset
yaya-benchmark -d <DATASET_PATH> --cfg_path <CFG_PATH> --weights_path <WEIGHTS_PATH> --names_path <NAMES_PATH> [--confidence <CONF>] [--nms_threshold <NMS>]

# Run inference on video
yaya-inference --video <INPUT_VIDEO> --output <OUTPUT_VIDEO> --cfg_path <CFG> --weights_path <WEIGHTS> --names_path <NAMES> [--confidence <CONF>] [--nms_threshold <NMS>] [--tracking] [--gpu]

# Augment dataset (selection + augmentation flags)
yaya-augument -i <DATASET_PATH> [selection flags] [augmentation flags] [-n <ITERATIONS>]

# GUI inference (Qt5)
yaya-inference-qt5

# GUI Darknet log comparison (Qt5)
yaya-darknet-logs-qt5 --log1 <LOG1_PATH> --log2 <LOG2_PATH>
```


