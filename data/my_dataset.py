import glob
import random
import os
import SimpleITK as sitk
import albumentations
import numpy as np
from data.base_dataset import BaseDataset
import albumentations as A
import torch
import torchvision.transforms as transform
from data.image_folder import make_dataset

class Mydataset(BaseDataset):

    def __init__(self, opt ): #phase="test"
        BaseDataset.__init__(self, opt)



        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        # self.files_A = sorted(glob.glob(self.dir_A + "/*.*"))      #找到这下面所有的文件
        # self.files_B = sorted(glob.glob(self.dir_B + "/*.*"))
        self.phase = opt.phase
        #self.mode = opt.mode
        if opt.phase == 'train':
            self.transformed = A.Compose(
                [

                    A.Resize(256,256),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    # A.Rotate(p=0.4),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=0.3),
                    A.OneOf([
                        A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                        #A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                        A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                    ], p=0.2),


                ]
            ,additional_targets = {'image2': 'image'}
            )
        else:
            self.transformed = A.Compose(
                [A.Resize(256, 256)],

                 additional_targets={'image2': 'image'}
            )
            print('I am here')


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        A_img = sitk.GetArrayFromImage(sitk.ReadImage(A_path))
        B_img = sitk.GetArrayFromImage(sitk.ReadImage(B_path))


        h1,w1 = A_img.shape
        h2,w2 = B_img.shape


        if h1 < w1:
            if (w1 - h1) % 2 == 0:
                A_img = np.pad(A_img, (((w1 - h1) // 2, (w1 - h1) // 2), (0, 0)), mode='reflect')   # oushu
            else:
                A_img = np.pad(A_img, (((w1 - h1) // 2, (w1 - h1) // 2 + 1), (0, 0)), mode='reflect')  #jishu
        else:
            if (h1 - w1) % 2 == 0:
                A_img = np.pad(A_img, ( (0, 0),((h1 - w1) // 2, (h1 - w1) // 2)), mode='reflect')
            else:
                A_img = np.pad(A_img, ((0, 0),((h1 - w1) // 2, (h1 - w1) // 2 + 1)), mode='reflect')

        if h2 < w2:
            if (w2 - h2) % 2 == 0:
                B_img = np.pad(B_img, (((w2 - h2) // 2, (w2 - h2) // 2), (0, 0)), mode='reflect')
            else:
                B_img = np.pad(B_img, (((w2 - h2) // 2, (w2 - h2) // 2 + 1), (0, 0)), mode='reflect')
        else:
            if (h2 - w2) % 2 == 0:
                B_img = np.pad(B_img, ( (0, 0),((h2 - w2) // 2, (h2 - w2) // 2)), mode='reflect')
            else:
                B_img = np.pad(B_img, ((0, 0),((h2 - w2) // 2, (h2 - w2) // 2 + 1)), mode='reflect')

        #print(A_img.shape,B_img.shape)







        dic_img = self.transformed(image=A_img,image2 = B_img)
        A = dic_img['image']
        B = dic_img['image2']

        A = torch.from_numpy(A/255).unsqueeze(0).float()
        B = torch.from_numpy(B/255).unsqueeze(0).float()
        A = A*2 - 1
        B = B*2 - 1




        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}



    def __len__(self):
        return max(self.A_size, self.B_size)
