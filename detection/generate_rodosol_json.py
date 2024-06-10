"""Generates rodosol .json file used for testing as ground truth."""
import argparse
import os
from generate_yolo_labels import read_file_as_dict, process_bb_string
import json
import tqdm

def generate_json(root_dir: str) -> None:
    split_file = os.path.join(root_dir, "split.txt")

    rodosol_gt = []
    img_paths = []

    # get paths to testing samples
    with open(split_file, 'r') as file:
        for line in tqdm.tqdm(file):
            file_path, label = line.strip().split(';')
            
            img_name = os.path.basename(file_path)
            img_path = os.path.normpath(os.path.join(root_dir, file_path))
            
            
            txt_file = f"{file_path.split('.')[1]}.txt"
            txt_full_path = os.path.join(root_dir, txt_file.lstrip(os.path.sep))
            
            if label != 'testing':
                continue
            img_paths.append(img_path)
            img_label = read_file_as_dict(txt_full_path)
            corners = process_bb_string(img_label["corners"])
            xyxy = [corners[0][0], corners[0][1], corners[2][0], corners[2][1]]


            gt = {"filename": img_name,
                  "dataset": "rodosol",
                  "objects":[
                  {"class_name": "plate",
                  "class_id": 0,
                  "box_type": "xyxy",
                  "box": xyxy}]}
            
            rodosol_gt.append(gt)

    print(f"Testing samples: {len(rodosol_gt)}")

    # save json
    save_path = "rodosol_gt.json"
    with open(save_path, "w") as fout:
        json.dump(rodosol_gt, fout, indent=2)

    # save .txt file
    with open("rodosol_test_samples.txt", 'w') as output:
        for row in img_paths:
            output.write(row + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rodosol_path",
        type=str,
        required=True,
        help="Path to rodosol dataset",
    )
    args=parser.parse_args()

    generate_json(args.rodosol_path)
