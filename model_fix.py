#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:26:10 2018

@author: apple
"""

# load model para
checkpoint = torch.load('./checkpoint/ckpt.t7')
# print(checkpoint['net'])
pretrain = checkpoint['net']

# print(pretrain['module.conv1.weight'].shape)
c1_w =  pretrain['module.conv1.weight'][:,0:1,:,:]
pretrain['module.conv1.weight'] = c1_w

# for k,v in pretrain.items():
# #     print(pretrain)
#     print(k,'\n')
#     print(v.shape,'\n')
#     print(pretrain)
# print(net.state_dict())
# for k,v in net.state_dict().items():
#     print(k,'\n')
#     print(v.shape,'\n')
pretrain = {k: v for k,v in pretrain.items() if k in net.state_dict()}
for k,v in pretrain.items():
#     print(pretrain)
    print(k,'\n')
    print(v.shape,'\n')
    
net.state_dict().update(pretrain)
# pretrain.update(net.)

net.load_state_dict(net.state_dict())
# net.load_state_dict(pretrain)
#     print(pretrain)
# pretrained_dict = ...
# model_dict = model.state_dict()

# # 1. filter out unnecessary keys
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # 2. overwrite entries in the existing state dict
# model_dict.update(pretrained_dict) 
# # 3. load the new state dict
# model.load_state_dict(pretrained_dict)
# model.load_state_dict(model_dict)