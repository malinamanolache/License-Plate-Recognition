import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
from collections import Counter
import bidict
import random
from scipy.ndimage import rotate

'''
def str_to_code(string, type, str_dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25,},
    digit_dict = {
    '0': 0, '1': 1, '2': 2, '3': 3,
    '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9}):
    if type == "br":
        adnotation = []
        adnotation.append(str_dict[string[0]])
        adnotation.append(str_dict[string[1]])
        adnotation.append(str_dict[string[2]])
        adnotation.append(digit_dict[string[3]])
        adnotation.append(digit_dict[string[4]])
        adnotation.append(digit_dict[string[5]])
        adnotation.append(digit_dict[string[6]])
        return adnotation
    if type == "me":
        adnotation = []
        adnotation.append(str_dict[string[0]])
        adnotation.append(str_dict[string[1]])
        adnotation.append(str_dict[string[2]])
        adnotation.append(digit_dict[string[3]])
        adnotation.append(str_dict[string[4]])
        adnotation.append(digit_dict[string[5]])
        adnotation.append(digit_dict[string[6]])
        return adnotation
    else:
        raise ValueError("Only 'br' and 'me' types are supported")
'''

def str_to_code(string, dict = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11,
    'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25,
    '0': 26, '1': 27, '2': 28, '3': 29,
    '4': 30, '5': 31, '6': 32, '7': 33,
    '8': 34, '9': 35
}):
    return [dict[char] for char in string]

class OcrDataset(Dataset):
    def __init__(self, json_file, dataset_type='rodosol'):
        """
        Initializes the dataset from a JSON file.

        :param json_file: Path to the JSON file containing the dataset annotations.
        :param dataset_type: One of "rodosol" or "ufpr"
        """
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Filter out entries that contain 'motorcycles' in the filename
        self.data = [entry for entry in self.data if 'motorcycles' not in entry['filename']]
        
        # Filter out entries with empty objects list
        self.data = [entry for entry in self.data if len(entry['objects']) > 0]
        
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)

    def _crop_image(self, image, box, box_type):
        """
        Crops the image based on the box and box type, ensuring the box is within image bounds.

        :param image: The original image.
        :param box: The bounding box.
        :param box_type: The type of the bounding box ('xyxy' or 'xywh').
        :return: The cropped image.
        """
        width, height = image.size
        
        if box_type == 'xyxy':
            left, top, right, bottom = box
        elif box_type == 'xywh':
            x, y, w, h = box
            left, top, right, bottom = x, y, x + w, y + h
        else:
            raise ValueError("Unsupported box type. Supported types are 'xyxy' and 'xywh'.")

        # Ensure the bounding box is within image bounds
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(0, min(right, width))
        bottom = max(0, min(bottom, height))

        return image.crop((left, top, right, bottom))

    def __getitem__(self, idx):
        """
        Returns the cropped image of the specified object with the highest confidence,
        along with the plate number and type.

        :param idx: Index of the item.
        :return: The cropped image tensor, label tensor, plate number, and plate type.
        """
        entry = self.data[idx]
        image_path = entry['filename']
        box_type = entry['box_type']

        # Find the object with the highest confidence
        objects = entry['objects']
        max_confidence_object = max(objects, key=lambda obj: obj['confidence'])
        box = max_confidence_object['box']

        image = Image.open(image_path).convert('RGB')
        cropped_image = self._crop_image(image, box, box_type)

        # Resize the cropped image if needed (you can adjust the size)
        cropped_image = cropped_image.resize((300, 150))
        image_array = np.array(cropped_image).astype(np.float32)
        image_array = image_array / 255.0  # Normalize to [0, 1]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Extract the plate number from the associated text file
        annotation_path = image_path.replace('.jpg', '.txt').replace('.png', '.txt')
        plate_number = self._load_plate_number(annotation_path)

        # Convert plate number to label tensor
        label_tensor = self._label_to_tensor(plate_number)
        
        # Determine plate type based on dataset type and file path
        plate_type = self._get_plate_type(image_path)
        
        return image_tensor, label_tensor, plate_number, plate_type

    def _load_plate_number(self, annotation_path):
        """
        Loads the plate number from the annotation file.

        :param annotation_path: Path to the annotation file.
        :return: The plate number as a string.
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            plate_info = {line.split(':')[0].strip(): line.split(':')[1].strip() for line in lines}
            plate_number = plate_info.get('plate', 'UNKNOWN').upper()

        return plate_number

    def _label_to_tensor(self, label):
        """
        Converts a string label to a one-hot encoded tensor.

        :param label: The string label to convert.
        :return: The one-hot encoded tensor.
        """
        classes_icxs = torch.tensor(np.array(str_to_code(label)))
        label_tensor = torch.zeros(len(classes_icxs), 36)
        label_tensor.scatter_(1, classes_icxs.type(torch.int64).unsqueeze(1), 1)
        return label_tensor

    def _get_plate_type(self, image_path):
        """
        Determines the plate type based on the dataset type and file path.

        :param image_path: The path to the image file.
        :return: The plate type as an integer.
        """
        if self.dataset_type == 'ufpr':
            return 1
        else:  # if rodosol
            if 'cars-br' in image_path:
                return 1
            elif 'cars-me' in image_path:
                return 0
            else:
                print(image_path)
                raise ValueError("image_path should contain 'cars-br' or 'cars-me'")



class OcrDataset_7chars_plateType(Dataset):
    def __init__(self, dataset_path, dataset_type='rodosol', split='train', augmentation_zaga=False, augmentation_overlap=False, augmentation_rotate=False, augmuemntation_level='low'):
        """
        Initializes the dataset for OCR task with images cropped to the license plates.

        :param dataset_path: Path to the dataset directory.
        :param split: One of 'training', 'testing', 'validation' to select the data split.
        :param dataset_type: One of "rodosol" or "ufpr"
        """
        self.dataset_path = dataset_path
        self.split = split
        self.data = []
        self.dataset_type = dataset_type
        self._load_split_data()
        self.augmentation_zaga, self.augmentation_overlap, self.augmentation_rotate = augmentation_zaga, augmentation_overlap, augmentation_rotate
        if augmuemntation_level == 'low':
            self.apply_augumentation_threshold = 0.5
            self.size_of_patch = 20
            self.number_of_patches = 4
            self.std = 0.1
        elif augmuemntation_level == 'high':
            self.apply_augumentation_threshold = 0.75
            self.size_of_patch = 25
            self.number_of_patches = 6
            self.std = 0.2

    def _load_split_data(self):
        """
        Loads the split data from the 'split.txt' file and filters it according to the specified split.
        """
        if self.dataset_type == 'rodosol':
            split_file_path = os.path.join(self.dataset_path, 'split.txt')
            with open(split_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    img_path, img_split = line.strip().split(';')
                    #print(img_path, img_split)
                    if img_split == self.split:
                        full_img_path = os.path.join(self.dataset_path, img_path.strip())
                        annotation_path = full_img_path.replace('.jpg', '.txt')
                        #print(full_img_path, annotation_path)
                        if os.path.exists(full_img_path) and os.path.exists(annotation_path):
                            self.data.append((full_img_path, annotation_path))
                            #print(self.data[-1])
        elif self.dataset_type == 'ufpr':
            split_dir_path = os.path.join(self.dataset_path, self.split)  # e.g., 'datasets/UFPR-ALPR/training'
            for dirpath, dirnames, filenames in os.walk(split_dir_path):
                for file in filenames:
                    if file.endswith('.png'):
                        image_path = os.path.join(dirpath, file)
                        annotation_path = image_path.replace('.png', '.txt')
                        if os.path.exists(annotation_path):
                            self.data.append((image_path, annotation_path))
        else:
            raise ValueError("dataset_type has to be 'rodosol' or 'ufpr'")

    def _load_image_label_type(self, image_path, annotation_path):
        """
        Loads the image and extracts the license plate as specified in the annotation file.
        """
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            plate_info = {line.split(':')[0].strip(): line.split(':')[1].strip() for line in lines}
            
            # This assumes 'corners' format is "x1,y1 x2,y2 x3,y3 x4,y4" without extra spaces around commas
            corners = plate_info['corners'].replace(' ', ',').split(',')
            corners = list(map(int, corners))  # Now converting to int should not raise an error

        # Extract corners
        x1, y1, x2, y2, x3, y3, x4, y4 = corners
        left, top, right, bottom = min(x1, x4), min(y1, y2), max(x2, x3), max(y3, y4)

        image = Image.open(image_path).convert('RGB')
        image = image.crop((left, top, right, bottom))
        img_resized = image.resize((300, 150))
        img_resized = np.array(img_resized)

        if self.dataset_type == 'ufpr':
            selected_type = 1
        else: # if rodosol
            if 'cars-br' in annotation_path:
                selected_type = 1
            elif 'cars-me' in annotation_path:
                selected_type = 0
            else:
                raise ValueError("annotation_path should contain 'cars-br' or 'cars-me'")

        return np.transpose(img_resized, (2, 0, 1)), plate_info['plate'], selected_type

    def _add_gaussian_noise(self, image):
        mean = 0
        #std = 0.1
        gaussian_noise = np.random.normal(mean, self.std, image.shape)
        noisy_image = image + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 1)  # Ensure the image is still in the valid range
        return noisy_image

    def _apply_random_overlap(self, image, size_of_patch):
        for _ in range(self.number_of_patches):  # Apply two random overlaps
            x = random.randint(0, image.shape[1] - size_of_patch)
            y = random.randint(0, image.shape[2] - size_of_patch)
            color = random.choice([0, 1])
            image[:, x:x+size_of_patch, y:y+size_of_patch] = color

        return image

    def _rotate_image(self, image):
        angle = random.uniform(-20, 20)
        rotated_image = np.empty_like(image)
        for i in range(image.shape[0]):
            rotated_image[i] = rotate(image[i], angle, reshape=False, mode='reflect')
        return rotated_image

    def _augment(self, image):
        if self.augmentation_zaga:
            #print('augmentation_zaga')
            #print(image)
            random_boolean = random.random() < self.apply_augumentation_threshold
            if random_boolean:
                image = self._add_gaussian_noise(image)
        if self.augmentation_overlap:
            #print('augmentation_overlap')
            #print(image)
            random_boolean = random.random() < self.apply_augumentation_threshold
            if random_boolean:
                image = self._apply_random_overlap(image, size_of_patch=self.size_of_patch)
        if self.augmentation_rotate:
            #print('_apply_random_overlap')
            #print(image)
            random_boolean = random.random() < self.apply_augumentation_threshold
            if random_boolean:
                image = self._rotate_image(image)
        #print(image)
        return image



    def __getitem__(self, idx):
        """
        Returns the cropped image of the license plate and the plate text in capital letters.
        """
        #print(self.data)
        image_path, annotation_path = self.data[idx]
        image, label, plate_type = self._load_image_label_type(image_path, annotation_path)
        label = label.upper()

        #classes_icxs = torch.tensor(string=np.array(str_to_code(label), type=plate_type))
        classes_icxs = torch.tensor(np.array(str_to_code(label)))
        #print(classes_icxs)
        label_tensor = torch.zeros(len(classes_icxs), 36)
        #print('llllllllllll')
        #print(classes_icxs.unsqueeze(1))
        label_tensor.scatter_(1, classes_icxs.type(torch.int64).unsqueeze(1), 1)
        #print(label_tensor.shape)
        #label_tensor[]
        image = image.astype(np.float32)
        image = (image - np.min(image))/(np.max(image) - np.min(image))
        #print('-----', image.shape)
        image = self._augment(image)
        return torch.from_numpy(image).float(), torch.from_numpy(np.array(label_tensor)), label, plate_type # **** be careful! At loss fucntion or nn output, the letters should also be capital

    def __len__(self):
        """
        Returns the total number of items in the dataset.
        """
        return len(self.data)

def display(image):
    print(np.transpose(image.numpy(), (1, 2, 0)).shape)
    image_pil = Image.fromarray((np.transpose(image.numpy(), (1, 2, 0))* 255).astype(np.uint8)) 
    image_pil.show()

if __name__ == '__main__':
    ####
    # Test ufpr
    ####
    #dataset = OcrDataset(r'datasets\UFPR-ALPR', dataset_type='ufpr', split='training') 
    ####
    # get statistics about the regions of the plates
    ####
    dataset = OcrDataset_7chars_plateType(r'datasets\RodoSol-ALPR', dataset_type='rodosol', split='training', augmentation_zaga=True, augmentation_overlap=True, augmentation_rotate=True) 
    list_of_numbers = []
    for image, label_tensor, label, plate_type in tqdm(dataset):
        if plate_type == 1:
            list_of_numbers.append(label[0:3])

    counter = Counter(list_of_numbers)

    # Print the count of each string
    print('rodosol')
    for string, count in counter.items():
        print(f"{string}: {count}")

    dataset = OcrDataset_7chars_plateType(r'datasets\UFPR-ALPR', dataset_type='ufpr', split='training')
    list_of_numbers = []
    for image, label_tensor, label, plate_type in tqdm(dataset):
        if plate_type == 1:
            list_of_numbers.append(label[0:3])

    counter = Counter(list_of_numbers)

    # Print the count of each string
    print('ufpr')
    for string, count in counter.items():
        print(f"{string}: {count}")

    dataset = OcrDataset_7chars_plateType(r'datasets\RodoSol-ALPR', dataset_type='rodosol', split='training', augmentation_zaga=True, augmentation_overlap=True, augmentation_rotate=True) 
            
    image, plate_number_tensor, plate_number, plate_type = dataset[2000]
    display(image)
    image, plate_number_tensor, plate_number, plate_type = dataset[100]
    display(image)
    image, plate_number_tensor, plate_number, plate_type = dataset[40]
    display(image)
    image, plate_number_tensor, plate_number, plate_type = dataset[3033]
    display(image)
    image, plate_number_tensor, plate_number, plate_type = dataset[209]
    display(image)
    image, plate_number_tensor, plate_number, plate_type = dataset[1900]
    display(image)

    print(plate_number_tensor.shape)
    lens = []
    '''
    print("Plate number:", plate_number)
    for image, label in dataset:
        print(image.shape)
    '''
    plates_list = []
    plate_types = []
    for image, label_tensor, label, plate_type in tqdm(dataset):
        plate_types.append(plate_type)
        plates_list.append(label)
        lens.append(len(label))
        #print(label)
        print(image.shape)
    print(np.unique(plate_types))
    combined_string = "".join(plates_list)
    character_counts = Counter(combined_string)
    sorted_characters_by_frequency = character_counts.most_common()
    for char, count in sorted_characters_by_frequency:
        print(f"Character: {char}, Frequency: {count}")

    ####
    # Test rodosol
    ####
    dataset = OcrDataset_7chars_plateType(r'datasets\UFPR-ALPR', dataset_type='ufpr', split='training')

    # Example to get an image and its plate number
    image, plate_number_tensor, plate_number, plate_type = dataset[0]
    #image, plate_number_tensor, plate_number, plate_type = Image.fromarray(np.transpose(image, (1, 2, 0)))
    image_pil = Image.fromarray((np.transpose(image.numpy(), (1, 2, 0))* 255).astype(np.uint8)) 
    image_pil.show()
    print("Plate number:", plate_number)

    '''
    for image, label in dataset:
        print(image.shape)
    '''
    plates_list = []
    plate_types = []
    for image, label_tensor, label, plate_type in tqdm(dataset):
        plate_types.append(plate_type)
        plates_list.append(label)
        lens.append(len(label))
        #print(label)
    print(np.unique(plate_types))
    print(np.unique(np.array(lens)))

    unique_characters = set()

    combined_string = "".join(plates_list)
    character_counts = Counter(combined_string)
    sorted_characters_by_frequency = character_counts.most_common()
    for char, count in sorted_characters_by_frequency:
        print(f"Character: {char}, Frequency: {count}")

    print(np.unique(np.array(lens)))



