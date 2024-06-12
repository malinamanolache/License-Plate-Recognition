'''
Now just add more trainable laeyrs: 
'''

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.transforms import Resize

class IndividualCharacterClassifier(nn.Module):
    def __init__(self, num_classes=36):
        super(IndividualCharacterClassifier, self).__init__()

        
        # Remove the fully connected layer
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        
        # A linear projection to map feature dimensions
        self.classifier1 = nn.Linear(2560, 64)
        self.classifier2 = nn.Linear(65, 36)

    def forward(self, x, plate_type):
        '''
        # plate_pype = 0 for letters, 1 for digits
        '''
        
        
        features = self.backbone(x)
        features = features.flatten(start_dim=1)
        x = self.classifier1(features)
        x = torch.cat((x, plate_type), axis=1) # 0 for letters, 1 for digits
        x = self.classifier2(x)

        return x

class PlateTypeSmallClassifier(nn.Module):
    def __init__(self):
        super(PlateTypeSmallClassifier, self).__init__()

        
        # Remove the fully connected layer
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        
        # A linear projection to map feature dimensions
        self.classifier = nn.Linear(2304, 1)

    def forward(self, x):
        
        features = self.backbone(x)
        features = features.flatten(start_dim=1)
        x = self.classifier(features)

        return x

# Example of creating and using the model
if __name__ == "__main__":
    # Assume input images of size (3, H, W)
    model = IndividualCharacterClassifier(num_classes=36).to('cuda')
    # Example input tensor
    #inputs = [torch.randn(1, 3, 100, 50).to('cuda'), torch.randn(1, 3, 100, 50).to('cuda'), torch.randn(1, 3, 100, 50).to('cuda'), torch.randn(1, 3, 100, 50).to('cuda'), torch.randn(1, 3, 100, 50).to('cuda'), torch.randn(1, 3, 100, 50).to('cuda'), torch.randn(1, 3, 100, 50).to('cuda')]
    inputs = torch.randn(1, 3, 80, 20).to('cuda')
    character_type = torch.zeros(1, 1).to('cuda')
    # Forward pass
    outputs = model(inputs, character_type)
    print(outputs)
    print(outputs.shape) 

    model = PlateTypeSmallClassifier().to('cuda')
    input_to_small_classifier = torch.randn(1, 3, 150, 300).to('cuda')
    outputs = model(input_to_small_classifier)
    print(outputs)
    print(outputs.shape) 