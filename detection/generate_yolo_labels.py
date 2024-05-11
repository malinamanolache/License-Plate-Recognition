import argparse
import os 


def generate_labels(dataset_name: str, path: str) -> None:
    if dataset_name not in ["RodoSol-ALPR", "UFPR"]:
        raise ValueError("Dataset name must be one of ['RodoSol-ALPR', 'UFPR']")

    images_path = os.path.join(path, dataset_name, "images")
    obj_types = sorted(os.listdir(images_path))

    for obj_type in obj_types:
        img_dir = os.path.join(images_path, obj_type)
        files = sorted(os.listdir(img_dir))
        file_names = list(map(lambda x: x.split('.')[0], files))
        file_names = set(file_names)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to generate the labels, either Rodosol or UFPR")
    parser.add_argument("--path", type=str, required=True, help="Path to dataset")
    args=parser.parse_args()

    generate_labels(args.dataset, args.path)