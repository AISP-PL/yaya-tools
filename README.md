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
usage: yaya-resize -i DATASET_PATH -o OUTPUT_PATH [--width WIDTH] [--height HEIGHT] [--keep_aspect_ratio] [--copy_annotations] [-h] [-v]

YAYa dataset management tool

options:
  -i DATASET_PATH, --dataset_path DATASET_PATH
                        Path to the dataset folder
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the output folder
  --width WIDTH         Width of the resized image
  --height HEIGHT       Height of the resized image
  --keep_aspect_ratio   Keep aspect ratio of the resized image
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

```bash
usage: yaya-inference --video VIDEO --output OUTPUT --cfg_path CFG_PATH --weights_path WEIGHTS_PATH --names_path NAMES_PATH [--confidence CONFIDENCE] [--nms_threshold NMS_THRESHOLD] [--tracking] [--gpu] [-h] [-v]

YAYa dataset management tool

options:
  --video VIDEO         Path to the video file
  --output OUTPUT       Path to the output video file
  --cfg_path CFG_PATH   Path to the configuration file
  --weights_path WEIGHTS_PATH
                        Path to the weights file
  --names_path NAMES_PATH
                        Path to the names file
  --confidence CONFIDENCE
                        Confidence threshold
  --nms_threshold NMS_THRESHOLD
                        NMS threshold
  --tracking            If set, the detector will use tracking with ByteSORT.
  --gpu                 If set, the detector will use the GPU for inference.
  -h, --help            Show this help message and exit.
  -v                    Show version and exit.
```

## YAYA augment

```bash
usage: yaya-augument -i DATASET_PATH [--select_negatives] [--select_class_id [SELECT_CLASS_ID]] [--select_equalize] [--select_horizontal] [--select_vertical] [--select_diagonal_right] [--select_diagonal_left] [--select_large]
                     [--select_tiny] [--select_crowded] [--select_spacious] [-n [ITERATIONS]] [--flip_horizontal] [--flip_vertical] [--crop [CROP]] [--rotate [ROTATE]] [--randrotate [RANDROTATE]] [--brighten] [--sharpen] [--darken]
                     [--clahe] [--equalize] [--colorshift] [--isonoise] [--gaussnoise] [--multi_noise] [--downsize_padding] [--compression] [--degrade] [--spatter] [--spatter_big] [--spatter_small] [--blackboxing [BLACKBOXING]]
                     [--snow] [--rain] [--fog] [--sunflare] [--blur] [--blur_delicate] [--flip] [-mb] [--grayscale] [--sepia] [-h] [-v]

YAYa dataset management tool

options:
  -i DATASET_PATH, --dataset_path DATASET_PATH
                        Path to the dataset folder
  --select_negatives    Select only negative images
  --select_class_id [SELECT_CLASS_ID]
                        Select class id
  --select_equalize     Select images to equalize dataset class representation
  --select_horizontal   Select horizontal line of detections
  --select_vertical     Select vertical line of detections
  --select_diagonal_right
                        Select diagonal right line of detections
  --select_diagonal_left
                        Select diagonal left line of detections
  --select_large        Select large annotations
  --select_tiny         Select small annotations
  --select_crowded      Select crowded scenes
  --select_spacious     Select spacious scenes
  -n [ITERATIONS], --iterations [ITERATIONS]
                        Maximum number of created images
  --flip_horizontal     Flip horizontal image.
  --flip_vertical       Flip vertical image.
  --crop [CROP]         Augument by random Crop image (for ex 640).
  --rotate [ROTATE]     Augument by direct degrees rotation (for ex 90).
  --randrotate [RANDROTATE]
                        Random rotation from -degrees to degrees.
  --brighten            Random make image brighten and adjust contrast.
  --sharpen             Random make image sharpen.
  --darken              Random make image darkne and adjust contrast.
  --clahe               Apply CLAHE to image.
  --equalize            Equalize image.
  --colorshift          Random color shift in image
  --isonoise            Random add iso noise to image. Caution!! Hard!
  --gaussnoise          Random add gauss noise.
  --multi_noise         Random multi gauss noise.
  --downsize_padding    Downsize with black padding.
  --compression         compression image quality.
  --degrade             Degrade image quality.
  --spatter             Spatter add.
  --spatter_big         Spatter add.
  --spatter_small       Spatter add.
  --blackboxing [BLACKBOXING]
                        Blackboxing HxH parts of image.
  --snow                Snow add.
  --rain                Rain add.
  --fog                 Fog add.
  --sunflare            Sunflare add.
  --blur                Blur image.
  --blur_delicate       Blur delicate image.
  --flip                Flip randomly image.
  -mb, --medianblur     Median blur image.
  --grayscale           Convert image to grayscale.
  --sepia               Convert image to sepia.
  -h, --help            Show this help message and exit.
  -v                    Show version and exit.

```

# YAYA GUI inference (Qt5)

yaya-inference With QT5 GUI desktop application for inference on video files.


# YAYA GUI Darknet log comparison (Qt5)



```bash
usage: yaya-darknet-logs-qt5 [-h] [--log1 LOG1] [--log2 LOG2]

Compare two Darknet logs and show previews in a Qt5 window.

options:
  -h, --help   show this help message and exit
  --log1 LOG1  Path to log 1 file
  --log2 LOG2  Path to log 2 file
```

