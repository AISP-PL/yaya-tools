#!python3
"""
Python 3 wrapper for identifying objects in images

Requires DLL compilation

Both the GPU and no-GPU version should be compiled; the no-GPU version should be renamed "yolo_cpp_dll_nogpu.dll".

On a GPU system, you can force CPU evaluation by any of:

- Set global variable DARKNET_FORCE_CPU to True
- Set environment variable CUDA_VISIBLE_DEVICES to -1
- Set environment variable "FORCE_CPU" to "true"
- Set environment variable "DARKNET_PATH" to path darknet lib .so (for Linux)

Directly viewing or returning bounding-boxed images requires scikit-image to be installed (`pip install scikit-image`)

Original *nix 2.7:
https://github.com/pjreddie/darknet/blob/0f110834f4e18b30d5f101bf8f1724c34b7b83db/python/darknet.py
Windows Python 2.7 version:
https://github.com/AlexeyAB/darknet/blob/fc496d52bf22a0bb257300d3c79be9cd80e722cb/build/darknet/x64/darknet.py

@author: Philip Kahn
@date: 20180503
"""
import logging
import os
import random
from ctypes import CDLL, POINTER, RTLD_GLOBAL, Structure, c_char_p, c_float, c_int, c_void_p, pointer

import numpy as np
import supervision as sv  # type :ignore


class BOX(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("w", c_float), ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [
        ("bbox", BOX),
        ("classes", c_int),
        ("prob", POINTER(c_float)),
        ("mask", POINTER(c_float)),
        ("objectness", c_float),
        ("sort_class", c_int),
        ("uc", POINTER(c_float)),
        ("points", c_int),
        ("embeddings", POINTER(c_float)),
        ("embedding_size", c_int),
        ("sim", c_float),
        ("track_id", c_int),
    ]


class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int), ("dets", POINTER(DETECTION))]


class IMAGE(Structure):
    _fields_ = [("w", c_int), ("h", c_int), ("c", c_int), ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int), ("names", POINTER(c_char_p))]


def network_width(net):
    return lib.network_width(net)


def network_height(net):
    return lib.network_height(net)


def bbox2points(bbox: tuple) -> tuple:
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = round(x - (w / 2))
    xmax = round(x + (w / 2))
    ymin = round(y - (h / 2))
    ymax = round(y + (h / 2))
    return xmin, ymin, xmax, ymax


def class_colors(names):
    """
    Create a dict with one random BGR color for each
    class name
    """
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for name in names}


def load_network(config_file, names_file, weights, batch_size=1):
    """
    load model description and weights from config files
    args:
        config_file (str): path to .cfg model file
        data_file (str): path to .data model file
        weights (str): path to weights
    returns:
        network: trained model
        class_names
        class_colors
    """
    network = load_net_custom(config_file.encode("ascii"), weights.encode("ascii"), 0, batch_size)
    with open(names_file, "r") as f:
        lines = f.readlines()
        class_names = [line.strip() for line in lines]

    colors = class_colors(class_names)
    return network, class_names, colors


def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print(
                "{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(
                    label, confidence, x, y, w, h
                )
            )
        else:
            print("{}: {}%".format(label, confidence))


def draw_boxes(image, detections, colors):
    import cv2

    for detection in detections:
        label, confidence, bbox = detection
        left, top, right, bottom = bbox
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(
            image,
            "{} [{:.2f}]".format(label, float(confidence)),
            (left + 2, top - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            image,
            "{} [{:.2f}]".format(label, float(confidence)),
            (left, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return image


def decode_detection(detections):
    """Decode detection - confidence to 0..100% and
    bbox points to rectangle."""
    decoded = []
    for label, confidence, bbox in detections:
        confidence = round(confidence * 100, 2)
        decoded.append((str(label), confidence, bbox2points(bbox)))
    return decoded


def remove_negatives(detections, class_names, num):
    """
    Remove all classes with 0% confidence within the detection
    """
    predictions = []
    for j in range(num):
        for idx, name in enumerate(class_names):
            if detections[j].prob[idx] > 0:
                bbox = detections[j].bbox
                bbox = (bbox.x, bbox.y, bbox.w, bbox.h)
                predictions.append((name, detections[j].prob[idx], (bbox)))
    return predictions


def all_nms(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.
    Do not differ on categories.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after n
            on-maximum suppression.

    Raises:
        AssertionError: If `iou_threshold` is not within the
            closed range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, " f"{iou_threshold} given."
    )
    rows, columns = predictions.shape

    # add column #5 - category filled with zeros for agnostic nms
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    # sort predictions column #4 - score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    ious = sv.box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        # drop detections with iou > iou_threshold and
        condition = iou > iou_threshold
        keep = keep & ~condition

    return keep[sort_index.argsort()]


def detections_to_ndarray(detections, class_names: list, num: int, confidence) -> np.ndarray:
    """
    Remove all classes with 0% confidence within the detection
    and changed format.
    """
    # Number of classes
    class_number = len(class_names)
    predictions: list[np.ndarray] = []
    for det in detections[:num]:
        # Check : objectness score
        if det.objectness <= confidence:
            continue

        # Check : Best class score > confidence
        conf_index = np.argmax(det.prob[0:class_number])
        conf = det.prob[conf_index]
        if conf <= confidence:
            continue

        # Detection : Found! Added to predictions
        x1x2y1y2 = bbox2points((det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h))
        predictions.append(np.array(x1x2y1y2 + (conf, conf_index)))

    return np.array(predictions)


def detect_image(
    network,
    class_names,
    image,
    imwidth=0,
    imheight=0,
    thresh=0.5,
    hier_thresh=0.5,
    nms=0.45,
):
    """
    Returns a list with highest confidence class and their bbox
    """
    # Get detections
    pnum = pointer(c_int(0))
    predict_image(network, image)
    detections = get_network_boxes(network, imwidth, imheight, thresh, hier_thresh, None, 0, pnum, 0)
    num = pnum[0]

    # Darknet NMS : Filter
    if nms:
        do_nms_sort(detections, num, len(class_names), nms)

    # Reformat : Convert to ndarray
    predictions = detections_to_ndarray(detections, class_names, num, thresh)
    free_detections(detections, num)

    return predictions


hasGPU = True
if os.name == "nt":
    cwd = os.path.dirname(__file__)
    os.environ["PATH"] = cwd + ";" + os.environ["PATH"]
    winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
    winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
    envKeys = []
    for k, _v in os.environ.items():
        envKeys.append(k)
    try:
        try:
            tmp = os.environ["FORCE_CPU"].lower()
            if tmp in ["1", "true", "yes", "on"]:
                raise ValueError("ForceCPU")
            print("Flag value {} not forcing CPU mode".format(tmp))
        except KeyError:
            # We never set the flag
            if "CUDA_VISIBLE_DEVICES" in envKeys:
                if int(os.environ["CUDA_VISIBLE_DEVICES"]) < 0:
                    raise ValueError("ForceCPU")
            try:
                if DARKNET_FORCE_CPU:  # type: ignore
                    raise ValueError("ForceCPU")
            except NameError as cpu_error:
                print(cpu_error)
        if not os.path.exists(winGPUdll):
            raise ValueError("NoDLL")
        lib = CDLL(winGPUdll, RTLD_GLOBAL)
    except (KeyError, ValueError):
        hasGPU = False
        if os.path.exists(winNoGPUdll):
            lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
            print("Notice: CPU-only mode")
        else:
            # Try the other way, in case no_gpu was compile but not renamed
            lib = CDLL(winGPUdll, RTLD_GLOBAL)
            print(
                "Environment variables indicated a CPU run, but we didn't find {}. Trying a GPU run anyway.".format(
                    winNoGPUdll
                )
            )
else:
    try:
        # Load darknet library
        lib = CDLL("libdarknet.so", RTLD_GLOBAL)
        lib.network_width.argtypes = [c_void_p]
        lib.network_width.restype = c_int
        lib.network_height.argtypes = [c_void_p]
        lib.network_height.restype = c_int

        copy_image_from_bytes = lib.copy_image_from_bytes
        copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        predict = lib.network_predict_ptr
        predict.argtypes = [c_void_p, POINTER(c_float)]
        predict.restype = POINTER(c_float)

        if hasGPU:
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]

        init_cpu = lib.init_cpu

        make_image = lib.make_image
        make_image.argtypes = [c_int, c_int, c_int]
        make_image.restype = IMAGE

        get_network_boxes = lib.get_network_boxes
        get_network_boxes.argtypes = [
            c_void_p,
            c_int,
            c_int,
            c_float,
            c_float,
            POINTER(c_int),
            c_int,
            POINTER(c_int),
            c_int,
        ]
        get_network_boxes.restype = POINTER(DETECTION)

        make_network_boxes = lib.make_network_boxes
        make_network_boxes.argtypes = [c_void_p]
        make_network_boxes.restype = POINTER(DETECTION)

        free_detections = lib.free_detections
        free_detections.argtypes = [POINTER(DETECTION), c_int]

        free_batch_detections = lib.free_batch_detections
        free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

        free_ptrs = lib.free_ptrs
        free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        network_predict = lib.network_predict_ptr
        network_predict.argtypes = [c_void_p, POINTER(c_float)]

        reset_rnn = lib.reset_rnn
        reset_rnn.argtypes = [c_void_p]

        load_net = lib.load_network
        load_net.argtypes = [c_char_p, c_char_p, c_int]
        load_net.restype = c_void_p

        load_net_custom = lib.load_network_custom
        load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        load_net_custom.restype = c_void_p

        free_network_ptr = lib.free_network_ptr
        free_network_ptr.argtypes = [c_void_p]
        free_network_ptr.restype = c_void_p

        do_nms_obj = lib.do_nms_obj
        do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        do_nms_sort = lib.do_nms_sort
        do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        free_image = lib.free_image
        free_image.argtypes = [IMAGE]

        letterbox_image = lib.letterbox_image
        letterbox_image.argtypes = [IMAGE, c_int, c_int]
        letterbox_image.restype = IMAGE

        load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        load_image = lib.load_image_color
        load_image.argtypes = [c_char_p, c_int, c_int]
        load_image.restype = IMAGE

        rgbgr_image = lib.rgbgr_image
        rgbgr_image.argtypes = [IMAGE]

        predict_image = lib.network_predict_image
        predict_image.argtypes = [c_void_p, IMAGE]
        predict_image.restype = POINTER(c_float)

        predict_image_letterbox = lib.network_predict_image_letterbox
        predict_image_letterbox.argtypes = [c_void_p, IMAGE]
        predict_image_letterbox.restype = POINTER(c_float)

        network_predict_batch = lib.network_predict_batch
        network_predict_batch.argtypes = [
            c_void_p,
            IMAGE,
            c_int,
            c_int,
            c_int,
            c_float,
            c_float,
            POINTER(c_int),
            c_int,
            c_int,
        ]
        network_predict_batch.restype = POINTER(DETNUMPAIR)

    except OSError:
        # Darknet library is missing, return from the module
        logging.warning("Darknet library is missing. Please compile the library to use it.")
