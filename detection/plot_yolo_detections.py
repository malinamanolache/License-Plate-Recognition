import argparse
import json
import os
import tqdm
import cv2

def plot_detections(json_file: str, dataset_path: str, out_path: str) -> None:
    file = open(json_file)
    results = json.load(file)
    
    for result in tqdm.tqdm(results):
        img_partial_path = os.sep.join(os.path.normpath(result["filename"]).split(os.sep)[-2:])
        img_path = os.path.join(dataset_path, img_partial_path)

        img = cv2.imread(img_path)
        height, width, _ = img.shape


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--json", type=str, required=True, help="JSON file with detections")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to 'images' dir (for Rodosol)")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the images.")
    args=parser.parse_args()

    plot_detections(args.json, args.dataset_path, args.save_path)
