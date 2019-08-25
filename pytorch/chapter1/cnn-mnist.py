import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import datetime

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # (channel, filters, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        # (28,28), (14, 14), (7,7), (3,3)
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)
        # self.softmax1 = F.softmax(dim=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net.to(device)
print(net)
print(device)


transform = transforms.Compose([transforms.ToTensor()])
x_train = datasets.MNIST(root='./data',
                         train=True,
                         download=True,
                         transform=transform)
x_test = datasets.MNIST(root='./data',
                        train=False,
                        download=True,
                        transform=transform)
print("Train dataset size:", len(x_train))
print("Test dataset size", len(x_test))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())


train_loader = torch.utils.data.DataLoader(x_train,
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=4)

test_loader = torch.utils.data.DataLoader(x_test,
                                          batch_size=128,
                                          shuffle=True,
                                          num_workers=4)
log_interval = len(train_loader) // 10
start_time = datetime.datetime.now()
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch + 1,
                  (i + 1) * len(inputs),
                  len(train_loader.dataset),
                  100. * (i + 1)/ len(train_loader),
                  running_loss / (1 + log_interval)))
            running_loss = 0.0

elapsed_time = datetime.datetime.now() - start_time
print("Elapsed time (train): %s" % elapsed_time)
# run a test loop
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = net(inputs)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #pred = outputs.data.max(1)[1]  # get the index of the max log-probability
        #correct += pred.eq(labels.data).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

#test_loss /= len(test_loader.dataset)
#print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, len(test_loader.dataset),
#        100. * correct / len(test_loader.dataset)))
