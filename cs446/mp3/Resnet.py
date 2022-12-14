import matplotlib.pyplot as plt
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
import random
import sklearn.datasets



class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block. It is the
                          parameter C in the handout hw3.pdf.
        """
        super(Block, self).__init__()
        """
        Write your code here.
        """
        self.model = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_channels),
        nn.ReLU(),
        nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_channels))
        

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        active = nn.ReLU()
        return active(x+self.model(x))

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        
        self.con1 = nn.Conv2d(1, num_channels, 3, stride=2, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(num_channels)
        self.re = nn.ReLU()
        self.max1 = nn.MaxPool2d(2)
        block = Block(num_channels)
        self.block1 = block
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(num_channels,num_classes)
    

    

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.con1(x)
        x = self.batch1(x)
 
        x = self.re(x)

        x = self.max1(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.linear(x.view(x.shape[0],x.shape[1]))  
        
        return x
        
#a = ResNet(5)
#b = a(torch.rand((100,1,5,5)))    
#print(b.shape)    
def plot_resnet_loss_1():
    """
    Train ResNet with different parameters C on digits data and draw the training
    error vs the test error curve. To make life easier, we provide you with the
    starter code to load the digits data and draw the figures with different
    parameters C. You do not need to modify the starter code and you only need
    to implement the training part. Train your algorithms for 4000 epochs with mini
    batch size = 128 and step size 0.1.
    """
    
    sk_digits = sklearn.datasets.load_digits()
    (X, Y) = (torch.tensor(sk_digits.data).type(torch.float), torch.tensor(sk_digits.target))
    Y = Y.type(torch.LongTensor)
    #print(X.shape, Y.shape, X.max(), X.min())
    X /= X.max()
    n = X.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    (X, Y) = ({'tr': X[perm[:n//2], ...], 'te': X[perm[n//2:], ...]}, {'tr':Y[perm[:n//2]], 'te':Y[perm[n//2:]]})
    mb_sz = 128
    stepsize = 0.1

    for (_, (net_s, num_channels)) in enumerate([
        ('ResNet_1', 1),
        ('ResNet_2', 2),
        ('ResNet_4', 4),
    ]):
        losses = { 'tr' : [], 'te' : [] }
        net =  ResNet(num_channels)
        optimizer = torch.optim.SGD(net.parameters(), lr=stepsize)
        for i in range(4000):
            idxs = random.sample(range(X['tr'].shape[0]), mb_sz)
            (x, y) = (X['tr'][idxs, ...], Y['tr'][idxs])
            x = x.view(x.shape[0], 1, 8, 8)
            
            """
            Write your code here.
            """
            optimizer.zero_grad()

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(net(x), y)
            
            with torch.no_grad():
                if (i + 1) % 25 == 0:
                    x = X['te']
                    x = x.view(x.shape[0], 1, 8, 8)
                    yhat2 = net(x)
                    loss2 = torch.nn.CrossEntropyLoss()(yhat2, Y['te'])
                    #print(f"{i} {loss:.3f} {loss2:.3f}")
                    losses['tr'].append(loss.detach())
                    losses['te'].append(loss2.detach())
            loss.backward() 

            optimizer.step()
        for s in ['tr', 'te']:
            plt.figure(1)
            plt.plot(range(len(losses[s])), losses[s],
                     label = f"{net_s} {s}")
                     
    plt.figure(1)
    plt.title("risk curves")
    plt.legend()
    plt.savefig('f1.pdf')
    plt.show()        
#plot_resnet_loss_1()

def plot_resnet_loss_2():
    """
    Train ResNet with parameter C = 64 on digits data and draw the training
    error vs the test error curve. To make life easier, we provide you with the
    starter code to load the digits data and draw the figures with C = 64.
    You do not need to modify the starter code and you only need
    to implement the training part. Train your algorithms for 4000 epochs with mini
    batch size = 128 and step size 0.1.
    """
    
    sk_digits = sklearn.datasets.load_digits()
    (X, Y) = (torch.tensor(sk_digits.data).type(torch.float), torch.tensor(sk_digits.target))
    Y = Y.type(torch.LongTensor)
    #print(X.shape, Y.shape, X.max(), X.min())
    X /= X.max()
    n = X.shape[0]
    perm = list(range(n))
    random.shuffle(perm)
    (X, Y) = ({'tr': X[perm[:n//2], ...], 'te': X[perm[n//2:], ...]}, {'tr':Y[perm[:n//2]], 'te':Y[perm[n//2:]]})
    mb_sz = 128
    stepsize = 0.1

    for (_, (net_s, num_channels)) in enumerate([
        ('ResNet_64', 64),
    ]):
        losses = { 'tr' : [], 'te' : [] }
        net =  ResNet(num_channels)

        optimizer = torch.optim.SGD(net.parameters(), lr=stepsize)
        for i in range(4000):
            idxs = random.sample(range(X['tr'].shape[0]), mb_sz)
            (x, y) = (X['tr'][idxs, ...], Y['tr'][idxs])
            x = x.view(x.shape[0], 1, 8, 8)
            
            """
            Write your code here.
            """
            optimizer.zero_grad()

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(net(x), y)
            loss.backward() 

            optimizer.step()
            
            with torch.no_grad():
                if (i + 1) % 25 == 0:
                    x = X['te']
                    x = x.view(x.shape[0], 1, 8, 8)
                    yhat2 = net(x)
                    loss2 = torch.nn.CrossEntropyLoss()(yhat2, Y['te'])
                    #print(f"{i} {loss:.3f} {loss2:.3f}")
                    losses['tr'].append(loss.detach())
                    losses['te'].append(loss2.detach())
        for s in ['tr', 'te']:
            plt.figure(1)
            plt.plot(range(len(losses[s])), losses[s],
                     label = f"{net_s} {s}")
                     
    plt.figure(1)
    plt.title("risk curves")
    plt.legend()
    plt.savefig('f2.pdf')
    plt.show()
plot_resnet_loss_2()