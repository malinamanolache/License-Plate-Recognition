import argparse
import json
import os
import torchmetrics
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion, CompleteIntersectionOverUnion, GeneralizedIntersectionOverUnion
import torch
import numpy as np
import tqdm

def load_json(path: str) -> list:
    file = open(path)
    data = json.load(file)
    return data

def map_by_filename(list_of_dicts: list) -> dict:
    mapped_dict = {}

    for entry in list_of_dicts:
        filename = entry["filename"]

        # check for absolute path - don't think it works for Windows
        if filename[0] == "/":
            filename = os.path.basename(filename)

        mapped_dict[filename] = entry

    return mapped_dict

def torchmetrics_dict(inference_dict: dict, preds=False) -> dict:
    result_dict = {}

    objects = inference_dict["objects"]
    num_objects = len(objects)

    boxes = torch.zeros(num_objects, 4)
    labels = torch.zeros(num_objects, dtype=torch.int)

    if preds:
        scores = torch.zeros(num_objects)

    for idx, obj in enumerate(objects):
        boxes[idx, :] = torch.tensor(obj["box"]).float()
        labels[idx] = int(obj["class_id"])
        if preds:
            scores[idx] = obj["confidence"]

    result_dict = {
        "boxes": boxes,
        "labels": labels
    }

    if preds:
        result_dict["scores"] = scores

    return result_dict


def compute_metrics(gt_json: str, pred_json: str) -> None:
    ground_truths = map_by_filename(load_json(gt_json))
    predictions = map_by_filename(load_json(pred_json))

    files = predictions.keys()

    map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    iou = IntersectionOverUnion(box_format='xyxy')
    ciou = CompleteIntersectionOverUnion(box_format='xyxy')
    giou = GeneralizedIntersectionOverUnion(box_format='xyxy')

    for file in tqdm.tqdm(files):
        gt = [torchmetrics_dict(ground_truths[file])]
        pred = [torchmetrics_dict(predictions[file], preds=True)]

        map.update(pred, gt)
        iou.update(pred, gt)
        ciou.update(pred, gt)
        giou.update(pred, gt)

    final_map = map.compute()
    final_iou = iou.compute()
    final_ciou = ciou.compute()
    final_giou = giou.compute()
    
    print(f"MaP = {final_map['map'].item()}")
    print(f"IoU = {final_iou['iou']}")
    print(f"cIoU = {final_ciou['ciou']}")
    print(f"gIoU = {final_giou['giou']}")

    

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="JSON file with ground truths",
    )
    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="JSON file with predictions",
    )
    args = parser.parse_args()

    compute_metrics(args.gt, args.pred)