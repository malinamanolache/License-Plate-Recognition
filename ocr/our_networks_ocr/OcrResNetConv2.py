'''
I want more to just generate exactly 7 characters, so I moved on to this impelentation: 
'''

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.transforms import Resize

class OcrResNetConv2(nn.Module):
    def __init__(self, num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet34', device='cuda'):
        super(OcrResNetConv2, self).__init__()

        self.lstm_input = lstm_input
        self.lstm_layers = lstm_layers
        self.device = device
        
        # Initialize ResNet model as the backbone
        if resnet_model == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif resnet_model == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        elif resnet_model == 'resnet50':
            self.backbone = resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported ResNet model: Choose from 'resnet18', 'resnet34', or 'resnet50'")
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze the backbone parameters (**** this could chancge in the future)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # Get the number of output channels from the last block
        self.out_ch = 512 if resnet_model in ['resnet18', 'resnet34'] else 2048  # ResNet-18 and 34 - 512, ResNet-50 - 2048

        self.plate_type_predictor = nn.Sequential(
            nn.Conv2d(self.out_ch, 32, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.Linear(768, 1)
        )

        # Assume resnet output is [batch, 512, H, W] for example
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 7))  # H to 1, W to 7
        # This will convert each of the 7 width positions to class probabilities
        self.number_classifier = nn.Conv1d(512, num_classes, kernel_size=1)
        #self.number_classifier = nn.Linear(512, num_classes)
        self.classifier2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3584, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        heads = [nn.Linear(32+1, num_classes) for _ in range(7)]
        self.heads = nn.Sequential(*heads)

        self.sig = nn.Sigmoid()
        
        # A linear projection to map feature dimensions
        #self.features2sequence = nn.Linear(self.out_ch, self.lstm_input)

    def forward(self, x):
        # Extract features from backbone
        #print(type(x))
        features = self.backbone(x)

        predicted_plate_type = self.plate_type_predictor(features)

        #print('1 ', features.shape, predicted_plate_type.shape)
        
        # Reduce feature map height to 1 and flatten features for LSTM
        features = self.adaptive_pool(features)
        features = features.squeeze(2)  # Remove the height dimension -> THIS IS HOW THE LEFT TO RIGHT ORDER IS PRESERVED
        #print('2 ', features.shape)

        predicted_plate_type_attention = torch.zeros(features.shape)
        predicted_plate_type_simoided = self.sig(predicted_plate_type)
        #print('3, ', predicted_plate_type_attention.shape)
        for batch in range(predicted_plate_type.shape[0]):
            #print('4 ', torch.full_like(features[batch], predicted_plate_type_simoided[batch].item()).shape)
            predicted_plate_type_attention[batch, :] = torch.full_like(features[batch], predicted_plate_type_simoided[batch].item())
        predicted_plate_type_attention = predicted_plate_type_attention.to(self.device)
        features +=  predicted_plate_type_attention # aici e critic
        #print('5: ', features.shape)

        #x = self.adaptive_pool(x)  # New shape: [batch, channels, 1, 7]
        #x = x.squeeze(2)  # Remove the height dimension, shape: [batch, channels, 7]
        # Apply classifier to predict classes for each of the 7 positions
        #print(features.shape)

        # V1) with convolution
        # x = self.number_classifier(features)  
        # return x.permute(0, 2, 1), predicted_plate_type

        # V2) with linear heads
        x = self.classifier2(features) 
        #print(x.shape)
        x = torch.cat((x, predicted_plate_type_simoided), dim=1)
        #print(x.shape)
        heads = torch.stack([head(x) for head in self.heads], dim=1)
        return heads, predicted_plate_type

        

# Example of creating and using the model
if __name__ == "__main__":
    # Assume input images of size (3, H, W)
    model = OcrResNetConv1(num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18', device='cpu') 
    # Example input tensor
    inputs = torch.randn(1, 3, 150, 300)
    # Forward pass
    outputs, output_plate = model(inputs)
    #print(outputs)
    print(outputs.shape) 