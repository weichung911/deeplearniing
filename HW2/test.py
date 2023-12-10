import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.datasets as datasets
import os
from PIL import Image
import torchvision.transforms as transforms
from utilities import *

LearningRate = 0.001
EPOCH = 20
BATCH = 128
SAVE_CSV = 'eval.csv'
TEST_DIR = './test/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
        transforms.Resize(256),
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
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
                                           )
print('==>>> total training batch number: {}'.format(len(train_loader)))

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_dataset = datasets.ImageFolder(root='./test/', transform=transform_train)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH, 
                                          shuffle=False, num_workers=0)

print('==>>> total testing batch number: {}'.format(len(test_loader)))
print('========================================')

#Local Response Normalization(LRN)
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

#AlexNet network
class myAlexNet(torch.nn.Module):
    def __init__(self, num_classes = NUM_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride = 2),
            LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
            LRN(local_size=5, alpha=1e-4, beta=0.75, ACROSS_CHANNELS=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride=2),
        )
        self.layer6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
        )
        self.layer7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.layer8 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        x = x.view(x.size()[0], -1)
        x = self.layer8(self.layer7(self.layer6(x)))
        return x
    
    def fit(self, train_dataloader, num_epochs=10, lr=0.001):
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(self.parameters(), lr=lr)

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
                if i % 39 == 0:
                    print('[%d, %d] loss: %.03f'
                          % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

    def name(self):
        return "AlexNet"

alexnet = myAlexNet().to(device)
history_alexnet = alexnet.fit(train_loader,EPOCH,LearningRate)
alexnet.load_state_dict(torch.load('alexnet_weights.pth'))
alexnet.eval()
with torch.no_grad():
        results = []
        nameid = []

        for ii, data in enumerate(test_loader):
            images, filename = data
            images = images.to(device)
            outputs = alexnet(images)
            _, predicted = torch.max(outputs.data, 1)
            results.extend(predicted)
            nameid.extend(filename)  

