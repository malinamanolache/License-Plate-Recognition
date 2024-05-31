import cv2
import argparse
import time
import glob
import numpy as np
import os
import json


def detect(
    model,
    filename: str,
    class_names: list,
    conf_thresh: float = 0.5,
    nms_thresh: float = 0.5,
) -> tuple:

    image = cv2.imread(filename)
    classes, scores, boxes = model.detect(image, conf_thresh, nms_thresh)
    color = (0, 0, 255)

    result_dict = {}
    objects = []
    result_dict["filename"] = filename

    for classid, score, box in zip(classes, scores, boxes):
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(image, box, color, 2)
        cv2.putText(
            image, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

        objects.append(
            {
                "class_id": int(classid),
                "class_name": class_names[classid],
                "box": box.tolist(),
                "confidence": float(score),
            }
        )

    result_dict["objects"] = objects

    return (image, result_dict)


def run_detector(
    model_path: str, input_path: str, image_size: list, save_path: str
) -> None:
    # process model paths
    weights = glob.glob(f"{model_path}/*.weights")[0]
    classes = glob.glob(f"{model_path}/*.txt")[0]
    cfg = glob.glob(f"{model_path}/*.cfg")[0]

    # get model
    net = cv2.dnn.readNet(weights, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=image_size, scale=1 / 255, swapRB=True)

    # get classes
    class_names = []
    with open(classes, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    os.makedirs(save_path, exist_ok=True)

    if os.path.isdir(input_path):
        files = sorted(os.listdir(input_path))
        results = []

        for file in files:
            img = cv2.imread(file)
            img, result = detect(model, img, class_names)
            results.append(result)

            image_name = os.path.basename(input_path)
            cv2.imwrite(os.path.join(save_path, image_name), img)

        json_path = os.path.join(save_path, "result.json")
        with open(json_path, "w") as fout:
            json.dump([result], fout, indent=2)

    elif os.path.isfile(input_path):
        img, result = detect(model, input_path, class_names)

        # save image and result
        image_name = os.path.basename(input_path)
        cv2.imwrite(os.path.join(save_path, image_name), img)

        json_path = os.path.join(save_path, "result.json")
        with open(json_path, "w") as fout:
            json.dump([result], fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory that contains model config, weights and file with class names.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images.",
    )
    parser.add_argument(
        "--image_size",
        type=tuple,
        required=False,
        default=(704, 704),
        help="Shape of the yolo training images, can be found in the yolo cfg file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=False,
        default="/yolo_detections",
        help="Path to output directory.",
    )
    args = parser.parse_args()

    print(args.input)
    run_detector(args.model_path, args.input, args.image_size, args.out)
