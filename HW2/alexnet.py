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
EPOCH = 10
BATCH = 168
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

#AlexNet network
class AlexNet(torch.nn.Module):
    def __init__(self, num_classes = NUM_classes):
        super(AlexNet,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride = 2),
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=2),
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3,3), stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x
    
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
    
alexnet = AlexNet(len(train_dataset.classes)).to(device)
history_alexnet = alexnet.fit(train_loader,EPOCH,LearningRate)
torch.save(alexnet.state_dict(), 'alexnet_weights.pth')
# alexnet.load_state_dict(torch.load('alexnet_weights.pth'))
# alexnet.eval()
with torch.no_grad():
        results = []
        nameid = []
        total = 0
        correct = 0
        for ii, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = alexnet(inputs)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            results.extend(predicted.cpu().numpy())
            nameid.extend(labels.cpu().numpy())
            total += labels.size(0) 
            # print(predicted,labels)
            correct += (predicted==labels).sum().item()
accuracy = correct/total
print(f"alexnet accuracy: {accuracy * 100:.2f}%")

plt.plot(history_alexnet, label='AlexNet Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
