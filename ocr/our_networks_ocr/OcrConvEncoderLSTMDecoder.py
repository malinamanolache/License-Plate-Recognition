
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.transforms import Resize


'''
class OcrConvEncoderLSTMDecoder(nn.Module):
    def __init__(self, num_classes=36, lstm_input=512, lstm_layers=2, hidden_dim=256):
        super(OcrConvEncoderLSTMDecoder, self).__init__()

        self.lstm_input = lstm_input
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        
        self.backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 7))  # H to 1, W to 7
        
        # LSTM decoder
        self.lstm = nn.LSTM(input_size=self.lstm_input, hidden_size=self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        features = self.adaptive_pool(features)
        features = features.squeeze(2)  
        features = features.permute(0, 2, 1) 

        lstm_out, _ = self.lstm(features)
        
        output = self.fc(lstm_out)  # shape: [batch, 7, num_classes]
        
        return output, torch.zeros(x.shape[0], 1).to('cuda')
'''

import torch
import torch.nn as nn

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, batch_first=True):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers
        self.lstm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_sizes[i - 1]
            output_dim = hidden_sizes[i]
            self.lstm_layers.append(nn.LSTM(input_dim, output_dim, num_layers=1, batch_first=batch_first))
    
    def forward(self, x):
        output = x
        for i in range(self.num_layers):
            output, _ = self.lstm_layers[i](output)
        return output, torch.zeros(x.shape[0], 1).to('cuda')


class OcrConvEncoderLSTMDecoder(nn.Module):
    def __init__(self, lstm_input=512, lstm_hidden_dims=[256, 128, 36]):
        super(OcrConvEncoderLSTMDecoder, self).__init__()

        self.lstm_input = lstm_input
        self.lstm_hidden_dims = lstm_hidden_dims
        
        self.backbone = resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 7))  # H to 1, W to 7
        
        # LSTM decoder
        self.lstm = CustomLSTM(input_size=self.lstm_input, hidden_sizes=self.lstm_hidden_dims, num_layers=len(self.lstm_hidden_dims), batch_first=True)
        
        
    def forward(self, x):
        features = self.backbone(x)
        
        features = self.adaptive_pool(features)
        features = features.squeeze(2)  
        features = features.permute(0, 2, 1) 

        lstm_out, _ = self.lstm(features)
        #print('len(lstm_out) ', len(lstm_out))
        #print('lstm_out.shape: ', lstm_out.shape)
                
        return lstm_out, torch.zeros(x.shape[0], 1).to('cuda')



# Example of creating and using the model
if __name__ == "__main__":
    # Assume input images of size (3, H, W)
    model = OcrConvEncoderLSTMDecoder(lstm_input=512, lstm_hidden_dims=[256, 128, 36]) 
    # Example input tensor
    inputs = torch.randn(1, 3, 100, 300)
    # Forward pass
    outputs, type = model(inputs)
    #print(outputs)
    print(outputs.shape) 