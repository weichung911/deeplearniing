import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import gzip
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim as optim
from tqdm import tqdm

LearningRate = 0.00001
EPOCH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = datasets.MNIST(root='data',train=True, download=True,transform=transform)
test_dataset = datasets.MNIST(root='data',train=False, download=True,transform=transform)

batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
labels = train_dataset.targets

# Print the first few labels
print(labels[:10])
print("Number of batches in the training loader:", len(train_loader))
print("Number of batches in the testing loader:", len(test_loader))


class VGG(nn.Module):
    """
    Standard PyTorch implementation of VGG. Pretrained imagenet model is used.
    """
    def __init__(self):
        super().__init__()
    
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),
            
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True),

            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

        # We need these for MaxUnpool operation
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        self.feature_maps = OrderedDict()
        self.pool_locs = OrderedDict()
        
    def forward(self, x):
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d):
                x, location = layer(x)
            else:
                x = layer(x)
        
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
    def fit(self, train_loader, num_epochs=10, lr=0.0001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss_history = [] 
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            train_data = range(len(train_loader))
            progress_bar = tqdm(total=len(train_data), desc='Training Progress')
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                output = self(inputs)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.update(1)
            average_loss = running_loss / len(train_loader.dataset)
            loss_history.append(average_loss)
            progress_bar.close()

            print('[Epoch %d] Average Loss: %.03f' % (epoch + 1, average_loss))

        return loss_history

vgg= VGG().to(device)
history_resnet = vgg.fit(train_loader,EPOCH,LearningRate)
torch.save(vgg.state_dict(), 'vgg_weights.pth')

with torch.no_grad():
        results = []
        nameid = []
        total = 0
        correct = 0
        for ii, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg(inputs)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            results.extend(predicted.cpu().numpy())
            nameid.extend(labels.cpu().numpy())
            total += labels.size(0) 
            # print(predicted,labels)
            correct += (predicted==labels).sum().item()
accuracy = correct/total
print(f"vgg accuracy: {accuracy * 100:.2f}%")
plt.plot(history_resnet, label='Vgg Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()