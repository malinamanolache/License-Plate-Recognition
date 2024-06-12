#from our_networks_ocr.OcrBaselineLstm2 import OcrBaselineLstm
from our_networks_ocr.OcrConvEncoderLSTMDecoder import OcrConvEncoderLSTMDecoder
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
dataset_dir = r'/home/radu/fac/ai2/proiect_ia/Our_project (2)(1)/datasets/rodosol'

batch_size = 32
#batch_size = 1
#batch_size = 8
epochs = 80
#lr = 1e-3
#lr = 1e-4
#lr = 1e-5
lr = 1e-6
l2_penalty = 1e-3

device = 'cuda'



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

#model = OcrBaselineLstm(num_classes=36) # train with 10^-4 90 epchs then 10^-5 another 10 epchs
#model = OcrConvEncoderLSTMDecoder(lstm_input=512, lstm_hidden_dims=[256, 128, 36]) # train with 10^-4 90 epchs then 10^-5 another 10 epchs
model = OcrConvEncoderLSTMDecoder(lstm_input=512, lstm_hidden_dims=[256, 36]) # train with 10^-4 90 epchs then 10^-5 another 10 epchs


optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalty)
criterion = nn.CrossEntropyLoss() 
loss_plate_type = nn.BCEWithLogitsLoss()


folder_path = "models/ocr/lstm_ocr"
#file_name = "OcrConvEncoderLSTMDecoder_256_128_36.pth"
file_name = "OcrConvEncoderLSTMDecoder_256_36.pth"

torch.autograd.set_detect_anomaly(True)

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