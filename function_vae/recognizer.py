import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import torch.nn as nn


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(kernel_size=5, in_channels=1, out_channels=16)
        self.conv2 = nn.Conv2d(kernel_size=5, in_channels=16, out_channels=32)

        self.fc = nn.Linear(in_features=512, out_features=10)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def recognition_digit(pil_img):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    cnnet = torch.load("function_vae/save_weights/cnnet")
    pil_img = pil_img.resize((28, 28)).convert('1')
    img_transformed = transform(pil_img)
    img_transformed = img_transformed.unsqueeze(0)
    with torch.no_grad():
        output = cnnet(img_transformed.to(torch.float32))
    _, predicted = torch.max(output.data, 1)
    return predicted.item()
