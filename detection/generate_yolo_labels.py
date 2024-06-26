import argparse
import os 
import json
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from collections import namedtuple
import cv2

YoloLabel = namedtuple('YoloLabel', ['x', 'y', 'w', 'h'])

def plot_processed_rodosol_label(image_path: str, label_path: str, out_path: str) -> None:
    """
    Plots processed RodoSol label in xywh format.

    Args:
        image_path: Path to RodoSol image sample.
        label_path: Path to corresponding RodoSol image label.
        out_path: Path to save the resulting plot.

    Returns:
        Saves plot with center bb maerked with "o" and middle of rectangle verticed marked with "X".

    Raises:
        ValueError: If image and label don't have the same name.
    """
    
    if image_path.split(".")[0] != label_path.split(".")[0]:
        raise ValueError("Image and label files must have the same name.")
    
    label_dict = read_file_as_dict(label_path)
    img = mpimg.imread(image_path)
    img_height, img_width, _ = img.shape
    
    bb_corners = np.array(process_bb_string(label_dict["corners"]))
    normalized = bb_corners_to_xywh(bb_corners, img_height, img_width)

    label = YoloLabel(x=int(normalized.x * img_width),
                       y=int(normalized.y * img_height),
                       w=int(normalized.w * img_width),
                       h=int(normalized.h * img_height))

    plt.figure()
    plt.imshow(img)
    plt.scatter(label.x, label.y, marker="o", color="red", s=10)
    plt.scatter(label.x, label.y - label.h//2, marker="x", color="red", s=70)
    plt.scatter(label.x, label.y + label.h//2, marker="x", color="red", s=70)
    plt.scatter(label.x - label.w//2, label.y, marker="x", color="red", s=70)
    plt.scatter(label.x + label.w//2, label.y, marker="x", color="red", s=70)

    plt.xticks([]), plt.yticks([])
    plt.savefig(out_path, bbox_inches='tight')
    plt.clf()
    plt.close()



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
    plt.savefig(out_path, bbox_inches='tight')
    plt.clf()
    plt.close()


def bb_corners_to_xywh(bb_corners: List[Tuple], img_height: int, img_width: int) -> YoloLabel:
    """
    Transforms bb corners to x, y, width, height normalized by the image width and height.

    Args:
        bb_corners: List of tuples with the corner coordinates, the order is:
                    top left, top right, bottom right, bottom left.
        img_height: Height of corresponding image.
        img_width: Width of corresponding img.
    
    Returns:
        Tuple with bb in xywh format.
    """
    top_left = bb_corners[0]
    top_right = bb_corners[1]
    bottom_left = bb_corners[2]
    bottom_right = bb_corners[3]

    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]
    x = top_left[0] + (width // 2)
    y = top_left[1] + (height // 2) 

    return YoloLabel(x=x/img_width, y=y/img_height, w=width/img_width, h=height/img_height)

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
    """
    Reads RodoSol .txt label file and returns a dict. Example of RodoSol label:
    
    type: car
    plate: ODE2510
    layout: Brazilian
    corners: 558,438 687,439 687,482 558,481


    Args:
        file_path: Path to .txt label file.

    Returns:
        Dict with 4 keys.

    """
    data = {}  
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into key and value using ':'
            key, value = line.strip().split(': ')
            # Add the key-value pair to the dictionary
            data[key] = value
    return data


def write_to_file(content: str, output_file: str) -> None:
    """
    Writes data to .txt file.

    Args:
        content: Data to be saved.
        output_file: Path to output file.
    """
    with open(output_file, 'a') as file:
        file.write(content + '\n')


def generate_train_test_valid_files(dataset_path: str, split_path: str) -> None:
    """
    Generates train.txt, valid.txt and test.txt files for yolo trainings. RodoSol
    provides a split.txt file containing the image paths and corresponding split as follows:

    ./images/cars-br/img_004765.jpg;validation
    ./images/cars-br/img_004150.jpg;training
    ./images/cars-me/img_013144.jpg;testing

    To the image paths, "data/obj/" is prepended to prepare the files for darknet.


    Args:
        split_path: Path to split.txt file from RodoSol.

    """
    train_file = "train_rodosol.txt"
    valid_file = "valid_rodosol.txt"
    test_file = "test_rodosol.txt"

    with open(split_path, 'r') as file:
        for line in file:
            file_path, label = line.strip().split(';')
            file_path = file_path.lstrip("./")
            file_path = os.path.join(dataset_path, file_path)

            if label == 'training':
                write_to_file(file_path, train_file)
            elif label == 'validation':
                write_to_file(file_path, valid_file)
            elif label == 'testing':
                write_to_file(file_path, test_file)

def save_yolo_label_to_file(path: str, label: YoloLabel, obj_class: int=0) -> None:
    """
    Saves yolo xywh label format to txt file

    Args:
        path: Path to .txt file.
        label: YoloLabel tuple.
        obj_class: The class of the object, defaults to 0 as RodoSol has 1 class only: license plate

    """
    with open(path, 'w') as file:
        file.write(f"{obj_class} {label.x} {label.y} {label.w} {label.h}\n")


def generate_labels(dataset_name: str, path: str) -> None:
    if dataset_name not in ["rodosol", "ufpr"]:
        raise ValueError("Dataset name must be one of ['rodosol', 'ufpr']")

    # create train test valid split files
    if dataset_name == "rodosol": 
        split_path = os.path.join(path, "split.txt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"RodoSol should have a split.txt file, check the dataset!")
        else:
            generate_train_test_valid_files(path, split_path)
        
        images_path = os.path.join(path, "images")
        obj_types = sorted(os.listdir(images_path))
        files = []

        for obj_type in obj_types:
            img_dir = os.path.join(images_path, obj_type)
            files = sorted(os.listdir(img_dir))

            # get the absolute path of the files
            abs_paths = list(map(lambda x: os.path.join(path, "images", obj_type, x), files))
            abs_paths = set(abs_paths)
            # remove extensions
            file_names = list(map(lambda x: x.split('.')[0], abs_paths))
            

            files += file_names
            print(f"{len(files)} samples in {dataset_name}")

    elif dataset_name == "ufpr":
        splits = os.listdir(path)
        files = []

        for split in splits:
            
            split_path = os.path.join(path, split)
            if not os.path.isdir(split_path):
                continue
            tracks = os.listdir(split_path)

            for track in tracks:
                track_path = os.path.join(path, split, track)
                track_files = os.listdir(track_path)
                abs_paths = list(map(lambda x: os.path.join(track_path, x), track_files))

                # remove extensions
                file_names = list(map(lambda x: x.split('.')[0], abs_paths))
                file_names = set(file_names)

                files += file_names

            paths_txt_file = f"{split}_ufpr.txt"
            img_paths = list(map(lambda x: f"{x}.png", files))

            # save paths to .txt file for yolo
            with open(paths_txt_file, 'w') as f:
                for line in img_paths:
                    f.write(f"{line}\n")

        print(f"{len(files)} samples in {dataset_name}")
    
    
    for filename in files:
        if dataset_name == "rodosol":
            img_format = "jpg"
        else:
            img_format = "png"
        
        image_path = f"{filename}.{img_format}"
        label_path = f"{filename}.txt"

        print(image_path)

        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape

        # read label file and extract bb
        label_dict = read_file_as_dict(label_path)
        label_dict["corners"] = process_bb_string(label_dict["corners"])

        # save original label to json to not lose info
        json_path = os.path.join(f"{filename}.json")
        with open(json_path, "w") as outfile: 
            json.dump(label_dict, outfile)

        yolo_label = bb_corners_to_xywh(label_dict["corners"], img_height, img_width)
        save_yolo_label_to_file(label_path, yolo_label)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to generate the labels, either Rodosol or UFPR")
    parser.add_argument("--path", type=str, required=True, help="Path to dataset")
    args=parser.parse_args()

    generate_labels(args.dataset, args.path)

    
    # Example on how to visualize the original and processed RodoSol plots:

    # test_img_path = os.path.join(args.path, "testing/track0091/track0091[01].png")
    # test_label_path = os.path.join(args.path, "testing/track0091/track0091[01].txt")
    
    # plot_rodosol_label(test_img_path, test_label_path, "original_label.png")
    # plot_processed_rodosol_label(test_img_path, test_label_path, "processed_label.png")
    