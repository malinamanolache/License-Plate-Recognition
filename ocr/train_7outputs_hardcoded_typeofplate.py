#from our_networks_ocr.OcrResNetConv1 import OcrResNetConv1
from our_networks_ocr.OcrResNetConv2 import OcrResNetConv2
#from our_networks_ocr.OcrResNetConv3 import OcrResNetConv3
#from our_networks_ocr.OcrResNetConvUnfreezed import OcrResNetConvUnfreezed
#from our_networks_ocr.OcrResNetConv3 import OcrResNetConv3
from our_networks_ocr.OcrConvNet1 import OcrConvMultiheadsNet
#from our_networks_ocr.OcrConvNet2 import OcrConvMultiheadsNet2
#from our_networks_ocr.OcrConvLstm import OcrConvLstm, OCRModel
from datasets import OcrDataset_7chars_plateType, str_to_code
import torch
import torch_utils_allinone
import torch.nn as nn


####
# Hyperparameters
####

#datset = 'ufpr'
dataset = 'rodosol'

#dataset_dir = r'datasets\UFPR-ALPR'
dataset_dir = r'datasets\RodoSol-ALPR'

batch_size = 32
epochs = 80
#lr = 1e-3
#lr = 1e-4
#lr = 1e-5
lr = 1e-6
l2_penalty = 1e-3

device = 'cuda'

'''
Random guessing cross entropy: 3.28 for 36 classes
'''

####
# Dataloaders
####

#train_dataset = OcrDataset_7chars_plateType(r'datasets\UFPR-ALPR', dataset_type='ufpr', split='training')
#val_dataset = OcrDataset_7chars_plateType(r'datasets\UFPR-ALPR', dataset_type='ufpr', split='validation')
#val_dataset = OcrDataset_7chars_plateType(r'datasets\UFPR-ALPR', dataset_type='ufpr', split='testing')
#test_dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='ufpr', split='testing')

#train_dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='training', augmentation_zaga=True, augmentation_overlap=True, augmentation_rotate=True, augmuemntation_level='low')
train_dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='training', augmentation_zaga=True, augmentation_overlap=True, augmentation_rotate=True, augmuemntation_level='high')
val_dataset = OcrDataset_7chars_plateType(dataset_dir, dataset_type='rodosol', split='validation', augmentation_zaga=False, augmentation_overlap=False, augmentation_rotate=False)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#model = OcrConvNet1(num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18')
#model = OcrConvMultiheadsNet(num_classes=36)
#model = OcrConvMultiheadsNet2(num_classes=36)
#model = OcrResNetConvUnfreezed(num_classes=36)
model = OcrResNetConv1(num_classes=36) # train with 10^-4 90 epchs then 10^-5 another 10 epchs
#model = OcrResNetConv2(num_classes=36)
#model = OcrResNetConv3(num_classes=36)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
criterion = nn.CrossEntropyLoss() 
loss_plate_type = nn.BCEWithLogitsLoss()

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# Define root folder and file_name for saving

#model = OcrResNetConv3(num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18')
# folder_path = "Our_project_3_classify/OCR/models/OcrResNetConv"
# file_name = "OcrResNetConv1_epch_100_copy.pth"

folder_path = "Our_project_3_classify/OCR/models/OcrResNetConv"
#file_name = "OcrResNetConv1_epch_100_plusaugmentation_low.pth"
#file_name = "OcrResNetConv1_epch_100_plusaugmentation_high.pth"
#file_name = "OcrResNetConv1_epch_100_plusaugmentation_high_10-6.pth"
file_name = "OcrResNetConv1_epch_100_plusaugmentation_high_10-6_ufprfinetuned.pth"
#file_name = "OcrResNetConv1_epch_100_plusaugmentation_high_reweightedclasses.pth"


# folder_path = "models/ocr/OcrConvNet"
# file_name = "OcrConvNet1_1.pth"

# folder_path = "models/ocr/OcrConvLstm"
# file_name = "OcrConvLstm_1.pth"

# cu reset 18 - mai bun
# folder_path = "models/ocr/OcrResNetConv2"
# file_name = "OcrResNetConv2.pth"

# cu resnet 34
# folder_path = "models/ocr/OcrResNetConv3"
# file_name = "OcrResNetConv3.pth"

# folder_path = "models/ocr/OcrConvMultiheadsNet2"
# file_name = "OcrConvMultiheadsNet2.pth"

train_losses, test_losses = torch_utils_allinone.train_loop(
                                                    model, 
                                                    train_loader=train_dataloader, 
                                                    optimizer=optimizer, 
                                                    loss=criterion, 
                                                    loss_fn_plate_type=loss_plate_type,
                                                    epochs=epochs, 
                                                    test_loader=val_dataloader, 
                                                    device=device, 
                                                    folder_path=folder_path, 
                                                    file_name=file_name,
                                                    print_frequency=1
                                                  )