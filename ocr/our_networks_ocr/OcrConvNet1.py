'''
Now just add more trainable laeyrs: 
'''

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.transforms import Resize

class OcrConvMultiheadsNet(nn.Module):
    def __init__(self, num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18'):
        super(OcrConvMultiheadsNet, self).__init__()

        self.lstm_input = lstm_input
        self.lstm_layers = lstm_layers
        
        # Remove the fully connected layer
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            #nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            #nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Dropout(0.3),
            #nn.BatchNorm2d(128),
        )

        # Assume resnet output is [batch, 512, H, W] for example
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 7))  # H to 1, W to 7
        # This will convert each of the 7 width positions to class probabilities
        self.classifier = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Conv1d(64, num_classes, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            #nn.BatchNorm1d(num_classes),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3456, 128),
            nn.ReLU(),
            #nn.Linear(128, 7*num_classes)
        )

        heads = [nn.Linear(128, num_classes) for _ in range(7)]
        self.heads = nn.Sequential(*heads)

        
        # A linear projection to map feature dimensions
        #self.features2sequence = nn.Linear(self.out_ch, self.lstm_input)

    def forward(self, x):
        # Extract features from backbone
        #print(type(x))
        features = self.backbone(x)
        
        # Reduce feature map height to 1 and flatten features for LSTM
        #features = self.adaptive_pool(features)
        #features = features.squeeze(2)  # Remove the height dimension -> THIS IS HOW THE LEFT TO RIGHT ORDER IS PRESERVED

        x = self.classifier(features)
        #print(self.heads[0](x).unsqueeze(1).shape)
        heads = torch.stack([head(x) for head in self.heads], dim=1)
        return heads
        #return x.permute(0, 2, 1)

# Example of creating and using the model
if __name__ == "__main__":
    # Assume input images of size (3, H, W)
    model = OcrConvMultiheadsNet(num_classes=36, lstm_input=512, lstm_layers=2, resnet_model='resnet18').to('cuda')
    # Example input tensor
    inputs = torch.randn(1, 3, 150, 300).to('cuda')
    # Forward pass
    outputs = model(inputs)
    print(outputs)
    print(outputs.shape) 