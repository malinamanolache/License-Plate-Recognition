import json
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch_utils_allinone

from torch import nn
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets import OcrDataset

device = 'cuda'

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




def display(image):
    print(np.transpose(image[0, :, :, :].numpy(), (1, 2, 0)).shape)
    image_pil = Image.fromarray((np.transpose(image[0, :, :, :].numpy(), (1, 2, 0))* 255).astype(np.uint8)) 
    image_pil.show()


# Example usage
if __name__ == "__main__":
    dataset_list = []
    json_paths = [r'git_IA2\License-Plate-Recognition\results\fasterRCNN_rodosol_pretrained_freezed-20240611T212819Z-001\fasterRCNN_rodosol_pretrained_freezed\result_rodosol.json', 
                  r'git_IA2\License-Plate-Recognition\results\fasterRCNN_rodosol_pretrained_freezed-20240611T212819Z-001\fasterRCNN_rodosol_pretrained_freezed\result_ufpr.json',
                  r'git_IA2\License-Plate-Recognition\results\fasterRCNN_rodosol_pretrained-20240611T212817Z-001\fasterRCNN_rodosol_pretrained\result_rodosol.json',
                  r'git_IA2\License-Plate-Recognition\results\fasterRCNN_rodosol_pretrained-20240611T212817Z-001\fasterRCNN_rodosol_pretrained\result_ufpr.json',
                  r'git_IA2\License-Plate-Recognition\results\fasterRCNN_rodosol_scratch-20240611T212820Z-001\fasterRCNN_rodosol_scratch\result_rodosol.json',
                  r'git_IA2\License-Plate-Recognition\results\fasterRCNN_rodosol_scratch-20240611T212820Z-001\fasterRCNN_rodosol_scratch\result_ufpr.json',
                  r'git_IA2\License-Plate-Recognition\results\yolov4_combined_pretrained-20240611T212806Z-001\yolov4_combined_pretrained\result_ufpr.json',
                  r'git_IA2\License-Plate-Recognition\results\yolov4_rodosol_pretrained-20240611T212811Z-001\yolov4_rodosol_pretrained\result_rodosol.json',
                  r'git_IA2\License-Plate-Recognition\results\yolov4_rodosol_pretrained-20240611T212811Z-001\yolov4_rodosol_pretrained\result_ufpr.json',
                  r'git_IA2\License-Plate-Recognition\results\yolov4_rodosol-20240611T212809Z-001\yolov4_rodosol\result_rodosol.json',
                  r'git_IA2\License-Plate-Recognition\results\yolov4_rodosol-20240611T212809Z-001\yolov4_rodosol\result_ufpr.json',
                  ]
    
    types = ['rodosol', 'ufpr', 'rodosol', 'ufpr', 'rodosol', 'ufpr', 'ufpr', 'rodosol', 'ufpr', 'rodosol', 'ufpr']
    
    for json_path, datset_type in zip(json_paths, types):
        dataset_list.append(OcrDataset(json_path, datset_type))

    ####### models
    folder_path = "."
    file_name = "Our_project_3_classify/OCR/models/OcrResNetConv/OcrResNetConv1_epch_100_plusaugmentation_high_10-5.pth"
    '''
    from our_networks_ocr.OcrResNetConv1 import OcrResNetConv1
    model = OcrResNetConv1(num_classes=36)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss() 
    loss_plate_type = nn.BCEWithLogitsLoss()
    '''
    from our_networks_ocr.OcrConvEncoderLSTMDecoder import OcrConvEncoderLSTMDecoder
    model = OcrConvEncoderLSTMDecoder(lstm_input=512, lstm_hidden_dims=[256, 128, 36]) # train with 10^-4 90 epchs then 10^-5 another 10 epchs


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss() 
    loss_plate_type = nn.BCEWithLogitsLoss()


    folder_path = "Our_project_4\our_networks_ocr"
    #file_name = "OcrConvEncoderLSTMDecoder_256_128_36.pth"
    file_name = "OcrConvEncoderLSTMDecoder_256_128_36.pth"
    ####### models

    i = 0
    for dataset in tqdm(dataset_list):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        i += 1
        print('INDEX: ', i)
        '''
        for image_tensor, label_tensor, plate_number, plate_type in dataloader:
            
            print(image_tensor.shape)  
            print(label_tensor.shape)  
            print(plate_type.shape)  
            print(plate_number)
            display(image_tensor)
        '''
        
        train_losses, test_losses = torch_utils_allinone.test_loop(
                                                            model, 
                                                            test_loader=dataloader,
                                                            optimizer=optimizer, 
                                                            loss=criterion, 
                                                            loss_fn_plate_type=loss_plate_type,
                                                            device=device, 
                                                            folder_path=folder_path, 
                                                            file_name=file_name,
                                                            print_frequency=1
                                                        )
