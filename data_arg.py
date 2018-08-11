
# coding: utf-8

# In[ ]:


import torch
import torchvision.transforms as transforms
import os
import matplotlib.pylab as plt
from PIL import Image


# In[ ]:


workspace = 'train_data'
label = 0


# In[ ]:


work_path = os.path.join(workspace,str(label))


# In[ ]:


transform1 = transforms.RandomHorizontalFlip(1)
transform2 = transforms.Compose([
    transforms.RandomVerticalFlip()
])
transform3 = transforms.RandomRotation(10)


# In[ ]:


for id in os.listdir(work_path):
    if id[0]=='.':
        continue
    path = os.path.join(work_path,id,'cpr')
#     print(path)
    for index,pic in enumerate(os.listdir(path),0):
        pic_path = os.path.join(path,pic)
        print(index,pic_path)
        img = Image.open(pic_path)
        img1 = transform1(img)
        img2 = transform2(img)
        img3 = transform3(img)
        img4 = transform3(img1)
        img5 = transform3(img2)
#         if index==0:
#             plt.imshow(img,cmap=plt.get_cmap('gray'))
#             plt.show()
# #             plt.tile('org')
#             print('arg')
#             plt.imshow(arg_pic,cmap=plt.get_cmap('gray'))
#             plt.show()
        outpath1 = pic_path.split('.')[0] + '_1.png'
#         outpath2 = pic_path.split('.')[0] + '_2.png'
#         outpath3 = pic_path.split('.')[0] + '_3.png'
#         outpath4 = pic_path.split('.')[0] + '_4.png'
#         outpath5 = pic_path.split('.')[0] + '_5.png'
        print(outpath1)
        img1.save(outpath1)
#         img2.save(outpath2)
#         img3.save(outpath3)
#         img4.save(outpath4)
#         img5.save(outpath5)

