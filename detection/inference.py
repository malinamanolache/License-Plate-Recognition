import cv2
import argparse
import time
import glob
import numpy as np
import os
import json
import torchvision
import torch
from torchvision import transforms
import tqdm



def detect_yolo(
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
    result_dict["model_name"] = "yolo"
    result_dict["box_type"] = "xywh"

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

def detect_fatserrcnn(model, filename: str, class_names: list) -> tuple:
    img = cv2.imread(str(filename))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    img_tensor = transform(img)
    img_tensor = img_tensor.to(device)
    preds = model([img_tensor])

    color = (0, 0, 255)

    result_dict = {}
    objects = []
    result_dict["filename"] = filename
    result_dict["model_name"] = "fasterRCNN"
    result_dict["box_type"] = "xyxy"

    # detach all values
    preds = [{k:v.cpu().detach().numpy() for k, v in preds[0].items()}]

    list_of_dicts = [dict(zip(preds[0],t)) for t in zip(*preds[0].values())]

    for pred in list_of_dicts:
        classid = pred["labels"] - 1 # for fastercnn 0 is background class
        score = pred["scores"]
        box = pred["boxes"]
        box = np.array(box, dtype=np.int64)
        label = "%s : %f" % (class_names[classid], score)
        
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(
            img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
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

    return (img, result_dict)


def detect(model_name: str, model, filename: str, save_path: str, draw_boxes: bool=False) -> list:
    class_names = ["plate"]
    
    if model_name == "yolo":
        img, result = detect_yolo(model, filename, class_names)
    elif model_name == "fasterrcnn":
        img, result = detect_fatserrcnn(model, filename, class_names)
    
    # save image with bounding boxes
    if draw_boxes:
        image_name = os.path.basename(filename)
        cv2.imwrite(os.path.join(save_path, image_name), img)

    return result

def run_detector(
    model_path: str, model_name: str, input_path: str, image_size: list, save_path: str, draw_boxes: bool=False
) -> None:
    
    os.makedirs(save_path, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # get model
    if model_name == "yolo":
        # process model paths
        weights = glob.glob(f"{model_path}/*.weights")[0]
        cfg = glob.glob(f"{model_path}/*.cfg")[0]

        # get model
        net = cv2.dnn.readNet(weights, cfg)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(size=image_size, scale=1 / 255, swapRB=True)
    
    elif model_name == "fasterrcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=None, 
                                                                num_classes=2)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()


    # process input
    if os.path.isdir(input_path):
        # assumes directory with images
        filenames = sorted(os.listdir(input_path))
        files = list(map(lambda x: os.path.join(input_path, x), filenames))
        
    elif os.path.isfile(input_path):
        # check if path is image
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            files = [input_path]
        
        elif input_path.lower().endswith('.txt'):
            # txt file with image paths
            files = []
            with open(input_path, 'r') as f:
                for line in f:
                    files.append(line.strip())
        else:
            raise ValueError("Input file should be either .png/.jpg/.jpeg image or .txt file with image paths.")

    results = []
    for file in tqdm.tqdm(files):
        result = detect(model_name, model, file, save_path, draw_boxes)
        results.append(result)
    
    json_path = os.path.join(save_path, "result.json")
    with open(json_path, "w") as fout:
        json.dump(results, fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="For yolo: directory that contains model config, weights and file with class names. \
            For fasterrcnn: path to pth file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolo", "fasterrcnn"],
        required=True,
        help="Model to perform inference on.",
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
        default="/result",
        help="Path to output directory.",
    )
    parser.add_argument(
        "--draw_bb",
        action='store_true',
        help="If passed it will save the images with the boundong boxes drawn."
    )

    args = parser.parse_args()

    run_detector(args.model_path, args.model, args.input, args.image_size, args.out, args.draw_bb)
