import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layer1 = nn.Sequential(
                        nn.Conv2d(3, 6, 5),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                      )
        conv_layer2 = nn.Sequential(
                        nn.Conv2d(6, 16, 5),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                      )
        self.conv_layers = nn.Sequential(
                            conv_layer1,
                            conv_layer2,
                            )
        self.fc_layer1 = nn.Sequential(
                            nn.Linear(16 * 5 * 5, 120),
                            nn.ReLU(inplace=True),
                        )
        self.fc_layer2 = nn.Sequential(
                            nn.Linear(120, 84),
                            nn.ReLU(inplace=True),
                            nn.Linear(84, 10),
                        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        return x
