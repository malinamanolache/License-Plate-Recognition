from torch.utils.data import Dataset
import os
import cv2
from torchvision import transforms
import torch

class RodosolDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir: str, split: str):

        if split not in ["training", "validation", "testing"]:
            raise ValueError("`subset` must be one of ['training', 'validation', 'testing']")
        
        self.root_dir = root_dir
        self.split = split

        self.file_paths = self._get_split_paths()

        # self.file_paths = self.file_paths[0:50]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # only one class - plate
        self.det_class = 1


    def _get_split_paths(self)-> list:
        filenames = []

        split_path = os.path.join(self.root_dir, "split.txt")
        with open(split_path, 'r') as file:
            for line in file:
                file_path, label = line.strip().split(';')

                if label == self.split:
                    absolute_path =  os.path.normpath(os.path.join(self.root_dir, file_path))
                    filenames.append(absolute_path)

        return filenames

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = cv2.imread(img_path)
        img = self.transform(img)

        label_path = os.path.splitext(self.file_paths[idx])[0] + ".txt"
        label_dict = self._read_file_as_dict(label_path)
        box = self._process_bb_string(label_dict["corners"])
        
        box_tensor = torch.tensor([box[0][0], box[0][1], box[2][0], box[2][1]], dtype=torch.float).unsqueeze(0)
        class_tensor = torch.tensor([self.det_class], dtype=torch.int64)

        box_width = box[2][0] - box[0][0]
        box_height = box[2][1] - box[0][1]
        area = torch.tensor([box_width * box_height], dtype=torch.float)

        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target = {
            "boxes": box_tensor,
            "labels": class_tensor,
            "image_id": idx,
            "iscrowd": iscrowd,
            "area": area,

        }

        return img, target

    def _read_file_as_dict(self, file_path: str) -> dict:
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

    def _process_bb_string(self, s: str) -> list[tuple]:
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