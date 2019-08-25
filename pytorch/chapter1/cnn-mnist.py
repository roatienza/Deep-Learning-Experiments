import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import datetime
import argparse

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # (channel, filters, kernel_size)
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        # (28,28), (13, 13), (6,6), (3,3)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    log_interval = len(train_loader) // 10
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if (i + 1) % log_interval == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch,
                  i * len(data),
                  len(train_loader.dataset),
                  100. * i / len(train_loader),
                  loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            test_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
          test_loss,
          correct,
          len(test_loader.dataset),
          100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    kwargs = {'num_workers': 4} if use_cuda else {}

    #transform = transforms.Compose([transforms.ToTensor(),
    #                                transforms.Normalize((0.1307,), (0.3081,))])
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


    DataLoader = torch.utils.data.DataLoader
    train_loader = DataLoader(x_train,
                              shuffle=True,
                              batch_size=args.batch_size,
                              **kwargs)

    test_loader = DataLoader(x_test,
                             shuffle=True,
                             batch_size=args.batch_size,
                             **kwargs)


    device = torch.device("cuda" if use_cuda else "cpu")
    model = Model().to(device)
    if torch.cuda.device_count() > 1:
        print("Available GPUs:", torch.cuda.device_count())
        # model = nn.DataParallel(model)
    print(model)
    print(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    start_time = datetime.datetime.now()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
    elapsed_time = datetime.datetime.now() - start_time
    print("Elapsed time (train): %s" % elapsed_time)
    test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
