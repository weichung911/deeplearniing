import torch 
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.version.cuda)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import gzip
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

LearningRate = 0.0001
EPOCH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfotm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='data',train=True, download=True,transform=transfotm)
test_dataset = datasets.MNIST(root='data',train=False, download=True,transform=transfotm)

batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
labels = train_dataset.targets

print(labels[:10])

print("Number of batches in the training loader:", len(train_loader))
print("Number of batches in the testing loader:", len(test_loader))

# CNN   
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * 28 * 28, 256) 
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def fit(self, train_loader, num_epochs=10, lr=0.0001):
        
        optimizer = torch.optim.Adam(self.parameters())
        error = nn.CrossEntropyLoss()
        loss_history = []
        
        for epoch in range(num_epochs):
            correct = 0
            running_loss = 0 
            train_data = range(len(train_loader))
            progress_bar = tqdm(total=len(train_data), desc="Training Progress")
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                var_X_batch = X_batch.to(device)
                var_y_batch = y_batch.to(device)
                optimizer.zero_grad()
                output = self(var_X_batch)
                loss = error(output, var_y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.update(1)
            average_loss = running_loss / len(train_loader.dataset)
            loss_history.append(average_loss)
            progress_bar.close()

            print('[Epoch %d] Average Loss: %.03f' % (epoch + 1, average_loss))
        return loss_history

cnn = CNN().to(device)
history_cnn = cnn.fit(train_loader,EPOCH,LearningRate)
torch.save(cnn.state_dict(), 'cnn_weights.pth')
# cnn.load_state_dict(torch.load('cnn_weights.pth'))
# cnn.eval()
with torch.no_grad():
        results = []
        nameid = []
        total = 0
        correct = 0
        
        for test_imgs, test_labels in test_loader:
        #print(test_imgs.shape)
            # test_imgs = Variable(test_imgs).float()
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
            output = cnn(test_imgs)
            predicted = torch.max(output,1)[1]
            total += test_labels.size(0)
            correct += (predicted == test_labels).sum()
accuracy = correct/total
print(f"cnn accuracy: {accuracy * 100:.2f}%")
plt.plot(history_cnn, label='cnn Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
                