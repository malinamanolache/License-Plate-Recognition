import argparse
import json
import os
import tqdm
import cv2
from typing import Dict, List, Tuple

def compute_corners_from_xywh(img_shape: Tuple, xywh: Tuple) -> List[Tuple]:
    height, width = img_shape

    center_x = int(xywh[0] * width)
    center_y = int(xywh[1] * height)

    bb_width = int(xywh[2] * width)
    bb_height = int(xywh[3] * height)

    top_left = (center_x - bb_width // 2, center_y - bb_height // 2)
    bottom_right = (center_x + bb_width // 2, center_y + bb_height // 2)

    return [top_left, bottom_right]


def plot_detections(json_file: str, dataset_path: str, out_path: str) -> None:
    file = open(json_file)
    results = json.load(file)
    
    for result in tqdm.tqdm(results):
        img_partial_path = os.sep.join(os.path.normpath(result["filename"]).split(os.sep)[-2:])
        img_path = os.path.join(dataset_path, img_partial_path)

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        boxes = []

        for obj in result["objects"]:
            relative_coordinates = obj["relative_coordinates"]
            xywh = (relative_coordinates["center_x"],
                    relative_coordinates["center_y"],
                    relative_coordinates["width"],
                    relative_coordinates["height"])

            corners = compute_corners_from_xywh((height, width), xywh)

            boxes.append(corners)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="JSON file with detections")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to 'images' dir (for Rodosol)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the images.")
    args=parser.parse_args()

    plot_detections(args.json, args.dataset_path, args.save_path)
