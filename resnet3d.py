
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
#torch.cuda.set_device(0)
# In[3]:


class Basicblock(nn.Module):
    def __init__(self,input,output,stride):
        super(Basicblock,self).__init__()
        self.conv1 =nn.Conv3d(input,output,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn = nn.BatchNorm3d(output)
        self.conv2 = nn.Conv3d(output,output,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.shortcut = nn.Sequential()
        if stride!=1:
            self.shortcut = nn.Sequential(nn.Conv3d(input,output,stride=stride,bias=False,
                                                 kernel_size=1),
                                          nn.BatchNorm3d(output)
                                         )
    def forward(self,x):
        out = self.conv1(x)
        print out.shape
        out = F.relu(self.bn(out))
        out = self.conv2(out)
        out = self.bn(out)
        print out.shape
        out += self.shortcut(x)
        out = F.relu(out)
        print out.shape
        return out
        


# In[55]:


class Resnet3d(nn.Module):
    def __init__(self,block,num_classes=2):
        super(Resnet3d,self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        print x.shape
#         print self.conv1(x).shape
        out = F.relu(self.bn1(self.conv1(x)))
        print out.shape
        out = self.layer1(out)      
        out = self.layer2(out)  
        out = self.layer3(out)   
        out = self.layer4(out)  
        out = F.avg_pool3d(out, 4)
        print(out.shape)
        out = out.view(out.size(0), -1)
        print(out.shape)
        out = self.linear(out)
        return out

        
        


# In[ ]:

device = "cuda" if torch.cuda.is_available() else 'cpu'
input = torch.randn(1,2,50,50,50)
net = Resnet3d(Basicblock)
net = net.to(device)
input = input.to(device)
output = net(input)
output.shape

