import numpy as np
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

from att_unet import AttU_Net as AttUnet
from torchvision import transforms

try:
    import cv2
except:
    print(f'cannot import cv2')
    pass


class model:
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.mean = None
        self.std = None
        
        self.model = AttUnet(img_ch=3, output_ch=3).cpu()

        self.tr_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])

        self.out_transforms = transforms.Compose([
            transforms.Resize((336, 544)),
        ])
    
    def load(self, path="./"):
        # self.model.load_from(config)
        model_path = ''
        for f_name in os.listdir(path):
            if f_name.startswith('model') and f_name.endswith('.pth'):
                model_path = os.path.join(path,f_name)
                print(f'found a model file {model_path}')

        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return self    

    def predict(self, X):
        """
        X: numpy array of shape (3,336,544)
        """
        print('submit unet attunet network')
        self.model.eval()
        # X = X / 255.0
        print(f'{X.shape}')
        print(f'X shape {X.shape}')

        X = np.transpose(X, axes=[1,2,0])

        image = self.tr_transforms(X) # image (3,224,224)
        image = image.unsqueeze(0) # image (1,3,224,224)

        seg,_ = self.model(image)  # seg (1,3,224,224)
        seg = torch.softmax(seg, dim=1) # seg (1,3,224,224)
        # seg = self.out_transforms(seg) # seg (1,3,336,544)

        seg = seg.squeeze(0).argmax(dim=0).detach().numpy()  # (224,224) values:{0,1,2} 1 upper 2 lower
        seg = cv2.resize(seg, (544,336), 0, 0, interpolation = cv2.INTER_NEAREST)
        seg = seg.astype(np.uint8)

        return seg

    def save(self, path="./"):
        '''
        Save a trained model.
        '''
        pass
