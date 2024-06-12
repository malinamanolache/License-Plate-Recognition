from our_networks_ocr.OcrResNetConv2 import OcrResNetConv2
#from our_networks_ocr.OcrResNetConvUnfreezed import OcrResNetConvUnfreezed
#from our_networks_ocr.OcrResNetConv3 import OcrResNetConv3
from our_networks_ocr.OcrConvNet1 import OcrConvMultiheadsNet
#from our_networks_ocr.OcrConvNet2 import OcrConvMultiheadsNet2
#from our_networks_ocr.OcrConvLstm import OcrConvLstm, OCRModel
from datasets import OcrDataset_7chars_plateType, str_to_code
import torch
import torch_utils_allinone
import torch.nn as nn
from PIL import Image
import numpy as np



####
# Hyperparameters
####

#datset = 'ufpr'
dataset = 'rodosol'

#dataset_dir = r'datasets\UFPR-ALPR'
dataset_dir = r'datasets\RodoSol-ALPR'

device = 'cuda'

'''
Random guessing cross entropy: 3.28 for 36 classes
'''

####
# Dataloaders
####

#train_dataset = OcrDataset(dataset_dir, dataset_type='ufpr', split='training')
#val_dataset = OcrDataset(dataset_dir, dataset_type='ufpr', split='validation')
#test_dataset = OcrDataset(dataset_dir, dataset_type='ufpr', split='testing')

train_dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='training')
val_dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='validation')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model = OcrConvNet1(num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18')
#model = OcrConvMultiheadsNet(num_classes=36)
#model = OcrConvMultiheadsNet2(num_classes=36)
model = OcrResNetConv2(num_classes=36)
model.eval()
#model = OcrResNetConv1(num_classes=36)

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Define root folder and file_name for saving

#model = OcrResNetConv3(num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18')
# folder_path = "models/ocr/OcrResNetConv"
# file_name = "OcrResNetConv1_1.pth"

# folder_path = "models/ocr/OcrConvNet"
# file_name = "OcrConvNet1_1.pth"

# folder_path = "models/ocr/OcrConvLstm"
# file_name = "OcrConvLstm_1.pth"


# folder_path = "models/ocr/OcrConvMultiheadsNet2"
# file_name = "OcrConvMultiheadsNet2.pth"

folder_path = "models/ocr/OcrResNetConvUnfreezed"
file_name = "OcrResNetConvUnfreezed.pth"

input, label, label_str = val_dataset[0]
output = model(input.unsqueeze(0))

image_pil = Image.fromarray((np.transpose(input.numpy(), (1, 2, 0))* 255).astype(np.uint8)) 
image_pil.show()
print('output: ', output)
print('label: ', label)
