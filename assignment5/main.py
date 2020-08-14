import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def loadData():
    transform = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset = torchvision.datasets.CIFAR10(root = './data',train = True,
                                            download=True,transform=transform)
    trainload = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainload,testloader,classes
def imshow(img):
    img = img/2 +0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
    
def main(epoch = 20):
    train,test,classes = loadData()
    net = Net().to(device)
    certerion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.9)
    runloss = 0
    losses = list()
    for i in range(epoch):
        for j,data in enumerate(train,0):
            inputs ,label = data
            inputs ,label = inputs.to(device),label.to(device)
            outputs = net(inputs)
            optimizer.zero_grad()
            loss = certerion(outputs,label)
            loss.backward()
            optimizer.step()
            runloss += loss.item()
            losses.append(loss.item())
            if j % 2000 == 1999:
                print('epoch:{},iterator{:5d},loss{:.5f}'.format(i,j,runloss/2000))
                runloss = 0
    print('finish training!')
    '''for data,j in enumerate(test,0):
            inputs ,label = data
            outputs = net(inputs)
            if outputs == label:'''
    torch.save(net,'NN.pth')
    return losses
    
if __name__=='__main__':
    loss = main(epoch = 20)
    with open('loss.txt','w') as file:
        for i in loss:
            file.write(str(i))
            file.write('\n')
