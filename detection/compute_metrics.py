import argparse
import json
import os

def load_json(path: str) -> list:
    file = open(path)
    data = json.load(file)
    return data

def build_dict(seq: list, key: str) -> dict:
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))

def compute_metrics(gt_json: str, pred_json: str) -> None:
    ground_truths = load_json(gt_json)
    predictions = load_json(pred_json)
    ground_truths = build_dict(ground_truths, key="filename")
    
    for pred in predictions:
        img_name = os.path.basename(pred["filename"])
        gt = ground_truths.get(img_name)

        print(pred)
        print(gt)

        exit()


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