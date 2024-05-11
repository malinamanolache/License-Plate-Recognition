import argparse
import os 
import json
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_rodosol_label(image_path: str, label_path: str, out_path: str) -> None:
    """
    Plots RodoSol bounding box corners. The purpose is to visualize the order of the corners.

    Args:
        image_path: Path to RodoSol image sample.
        label_path: Path to corresponding RodoSol image label.
        out_path: Path to save the resulting plot.

    Returns:
        Saves plot with corners marked with "X" and their order.

    Raises:
        ValueError: If image and label don't have the same name.
    """
    
    if image_path.split(".")[0] != label_path.split(".")[0]:
        raise ValueError("Image and label files must have the same name.")
    
    label_dict = read_file_as_dict(label_path)
    bb_corners = np.array(process_bb_string(label_dict["corners"]))
    img = mpimg.imread(image_path)
    
    plt.figure()
    plt.imshow(img)
    for idx, point in enumerate(bb_corners):
        # plot bb corner and order to figure how to process them later
        plt.scatter(point[0], point[1], marker="x", color="red", s=100)
        plt.text(point[0]-5, point[1], str(idx), fontsize="large", color="k", fontweight="bold")
    
    plt.xticks([]), plt.yticks([])
    plt.savefig(out_path)
    plt.clf()
    plt.close()



def process_bb_string(s: str) -> List[Tuple]:
    """
    Processes the boundind box string from RodoSol dataset

    Args:
        s: BB string defined as: "x,y x,y x,y x,y". Corners are 
           separated by spaces and coordinate values by commas.

    Returns:
        A list of 4 lements where each element is a tuple with the 
        corner coordinates

    """
    result = []
    corners = s.split()
    assert len(corners) == 4, "Bounding box does not match expected format. \
                               There should be 4 coordinate pairs separated by whitespaces."

    for c in corners:
        coordinates_str = c.split(",")
        assert len(coordinates_str) == 2, "Each corner must have 2 coordinates."
        # make them integers
        coordinates_int = tuple(map(lambda x: int(x), coordinates_str))
        result.append(coordinates_int)
    
    return result


def read_file_as_dict(file_path: str) -> Dict:
    data = {}  
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into key and value using ':'
            key, value = line.strip().split(': ')
            # Add the key-value pair to the dictionary
            data[key] = value
    return data


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

        for name in file_names:
            image_path = os.path.join(img_dir, f"{name}.jpg")
            label_path = os.path.join(img_dir, f"{name}.txt")

            # read label file and extract bb
            label_dict = read_file_as_dict(label_path)
            bb_corners = process_bb_string(label_dict["corners"])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to generate the labels, either Rodosol or UFPR")
    parser.add_argument("--path", type=str, required=True, help="Path to dataset")
    args=parser.parse_args()

    generate_labels(args.dataset, args.path)

    plot_rodosol_label("/home/ia2/datasets/RodoSol-ALPR/images/cars-br/img_000001.jpg", 
                        "/home/ia2/datasets/RodoSol-ALPR/images/cars-br/img_000002.txt",
                        "/home/ia2/datasets/RodoSol-ALPR/label.png")