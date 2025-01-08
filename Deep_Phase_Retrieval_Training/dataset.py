import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, load_npy, Augment_RGB_torch
import torch.nn.functional as F
import random

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, foc_list, num_training_samples, num_zernikes):
        super(DataLoaderTrain, self).__init__()
        gt_dir = 'Output_Data'
        input_dir = 'Output_Data'
        
        files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        caustics=[]
        for i in range(len(foc_list)):
            caustics.append([])
        zernike_files = []
        for i in range(int(len(files)/8)):
            pos=0
            for j in range(len(foc_list)):
                caustics[pos].append(files[8*i+foc_list[j]+3])
                pos+=1
            zernike_files.append(files[8*i+7])
        
        self.zernike_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in zernike_files]
        self.caustic_names=[]
        for i in range(len(foc_list)):
            self.caustic_names.append([])
        for i in range(len(self.caustic_names)):
            self.caustic_names[i] = [os.path.join(rgb_dir, input_dir, x) 
                                     for x in caustics[i] if is_png_file(x)]

        self.tar_size = num_training_samples  # get the size of target
        self.num_z=num_zernikes

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        zernike = torch.from_numpy(np.float32(load_npy(self.zernike_filenames[index])[3:3 + self.num_z]))
        #print(self.zernike_filenames[index])
        inp=()
        for i in range(len(self.caustic_names)):
            img = torch.from_numpy(np.float32(load_img(self.caustic_names[i][index])))
            inp = inp + (img, )
            #print(self.caustic_names[i][index])
        inp_stack = torch.stack(inp)

        return zernike, inp_stack

##################################################################################################
##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, foc_list, num_zernikes):
        super(DataLoaderVal, self).__init__()
        gt_dir = 'Val_Data'
        input_dir = 'Val_Data'
        
        files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        caustics=[]
        for i in range(len(foc_list)):
            caustics.append([])
        zernike_files = []
        for i in range(int(len(files)/8)):
            pos=0
            for j in range(len(foc_list)):
                caustics[pos].append(files[8*i+foc_list[j]+3])
                pos+=1
            zernike_files.append(files[8*i+7])
        
        self.zernike_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in zernike_files]
        self.caustic_names=[]
        for i in range(len(foc_list)):
            self.caustic_names.append([])
        for i in range(len(self.caustic_names)):
            self.caustic_names[i] = [os.path.join(rgb_dir, input_dir, x) 
                                     for x in caustics[i] if is_png_file(x)]

        self.tar_size = len(self.zernike_filenames)  # get the size of target
        self.num_z=num_zernikes

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        zernike = torch.from_numpy(np.float32(load_npy(self.zernike_filenames[index])[3:3 + self.num_z]))
        #print(self.zernike_filenames[index])
        inp=()
        for i in range(len(self.caustic_names)):
            img = torch.from_numpy(np.float32(load_img(self.caustic_names[i][index])))
            inp = inp + (img, )
            #print(self.caustic_names[i][index])
        inp_stack = torch.stack(inp)

        return zernike, inp_stack

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, foc_list):
        super(DataLoaderTest, self).__init__()
        gt_dir = 'Val_Data'
        input_dir = 'Val_Data'
        
        files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        caustics=[]
        for i in range(len(foc_list)):
            caustics.append([])
        zernike_files = []
        for i in range(int(len(files)/8)):
            pos=0
            for j in range(len(foc_list)):
                caustics[pos].append(files[8*i+foc_list[j]+3])
                pos+=1
            zernike_files.append(files[8*i+7])
        
        self.zernike_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in zernike_files]
        self.caustic_names=[]
        for i in range(len(foc_list)):
            self.caustic_names.append([])
        for i in range(len(self.caustic_names)):
            self.caustic_names[i] = [os.path.join(rgb_dir, input_dir, x) 
                                     for x in caustics[i] if is_png_file(x)]

        self.tar_size = len(self.zernike_filenames)  # get the size of target

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        zernike = torch.from_numpy(np.float32(load_npy(self.zernike_filenames[index])[3:]))
        #print(self.zernike_filenames[index])
        inp=()
        for i in range(len(self.caustic_names)):
            img = torch.from_numpy(np.float32(load_img(self.caustic_names[i][index])))
            inp = inp + (img, )
            #print(self.caustic_names[i][index])
        inp_stack = torch.stack(inp)

        return zernike, inp_stack