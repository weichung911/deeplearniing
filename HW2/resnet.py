import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.datasets as datasets
import os
from PIL import Image
import torchvision.transforms as transforms
from utilities import *
import matplotlib.pyplot as plt

LearningRate = 0.001
EPOCH = 20
BATCH = 16
TEST_DIR = './test/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
transform_train = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = datasets.ImageFolder(root='./train/', transform=transform_train)
print(train_dataset.class_to_idx)

NUM_classes = len(train_dataset.classes)
print('number of class: %d' %NUM_classes)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=BATCH,
                                           num_workers=0,
                                           pin_memory=True,
                                           shuffle=True
                                           )
print('==>>> total training batch number: {}'.format(len(train_loader)))

transform_test = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_dataset = datasets.ImageFolder(root='./test/', transform=transform_test)
true_labels = test_dataset.targets

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, 
                                          shuffle=False, num_workers=0)

print('==>>> total testing batch number: {}'.format(len(test_loader)))
print('========================================')
# ResNet50

class ConvBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, enable_relu: bool = True):
        super(ConvBlock, self).__init__()
        self.enable_relu = enable_relu
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        
        return F.relu(y, inplace=True) if self.enable_relu else y

class ResBlock(nn.Module):
    
    def __init__(self, input_channel: int, block_base_channel: int):
        super(ResBlock, self).__init__()
        equal_channel_size = input_channel == block_base_channel * 4
    
        self.block = nn.Sequential(
            ConvBlock(input_channel, block_base_channel, 1),
            ConvBlock(block_base_channel, block_base_channel, 3, padding = 1),
            ConvBlock(block_base_channel, block_base_channel * 4, 1)
        )
        
        self.downsample = nn.Identity() if equal_channel_size else nn.Sequential(
            ConvBlock(input_channel, block_base_channel * 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.downsample(x)

class ResNet50(nn.Module):
    
    def __init__(self):
        super(ResNet50, self).__init__()
        self.foot = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding = 3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2)
        )
        self.block1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(256, 64),
            ResBlock(256, 64)
        )
        self.block2 = nn.Sequential(
            ResBlock(256,128),
            ResBlock(512,128),
            ResBlock(512,128),
            ResBlock(512,128),
            nn.MaxPool2d(2,2)
        )
        self.block3 = nn.Sequential(
            ResBlock(512,256),
            ResBlock(1024,256),
            ResBlock(1024,256),
            ResBlock(1024,256),
            ResBlock(1024,256),
            ResBlock(1024,256),
            nn.MaxPool2d(2,2)
        )
        self.block4 = nn.Sequential(
            ResBlock(1024,512),
            ResBlock(2048,512),
            ResBlock(2048,512),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(2048,2)
        )

        
    def forward(self, x):
        tmp = self.foot(x)
        tmp = self.block1(tmp)
        tmp = self.block2(tmp)
        tmp = self.block3(tmp)
        tmp = self.block4(tmp)
        tmp = torch.flatten(tmp,start_dim=1)
        tmp = self.fc(tmp)

        return tmp
    
    def fit(self, train_loader, num_epochs=10, lr=0.0001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_history = [] 

        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                output = self(inputs)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            average_loss = running_loss / len(train_loader.dataset)
            loss_history.append(average_loss)

            print('[Epoch %d] Average Loss: %.03f' % (epoch + 1, average_loss))

        return loss_history
    
resnet50 = ResNet50().to(device)
history_resnet = resnet50.fit(train_loader,EPOCH,LearningRate)
torch.save(resnet50.state_dict(), 'resnet_weights.pth')
# resnet50.load_state_dict(torch.load('resnet_weights.pth'))
# resnet50.eval()

with torch.no_grad():
        results = []
        nameid = []
        total = 0
        correct = 0
        for ii, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet50(inputs)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            results.extend(predicted.cpu().numpy())
            nameid.extend(labels.cpu().numpy())
            total += labels.size(0) 
            # print(predicted,labels)
            correct += (predicted==labels).sum().item()
accuracy = correct/total
print(f"Resnet accuracy: {accuracy * 100:.2f}%")
plt.plot(history_resnet, label='Resnet Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()