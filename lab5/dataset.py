import torch
import os
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

default_transform = transforms.Compose([
    transforms.ToTensor()
    ])

class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform, seq_len=20, image_size=64):
        assert mode == 'train' or mode == 'test' or mode == 'validate'

        self.data_root = args.data_root
        self.transform = transform

        if mode == 'train':
            self.data_dir = self.data_root + '/train'
            self.ordered = False
        elif mode == 'test': 
            self.data_dir = self.data_root + '/test'
            self.ordered = True 
        else:
            self.data_dir = self.data_root + '/validate'
            self.ordered = True 
        
        self.data_dirs = []

        for folder_1 in os.listdir(self.data_dir):
            path = self.data_dir + '/' + folder_1
            for folder_2 in os.listdir(path):
                self.data_dirs.append(path + '/' + folder_2)
        
        self.len = len(self.data_dirs)
        self.seq_len = seq_len
        self.seed_is_set = False
        self.d = 0
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return self.len
        
    def get_seq(self, path):
        seq = []
        for i in range(self.seq_len):
            file = path + '/' + str(i) + '.png'
            img = cv2.imread(file)
            img = self.transform(img)
            seq.append(img)
        seq = torch.stack(seq)
        return seq

    
    def get_csv(self, path):
        action_file = path + '/actions.csv' 
        endeffector_positions_file = path + '/endeffector_positions.csv'

        with open(action_file, 'r') as f:
            action = list(csv.reader(f, delimiter=","))
            action = np.array(action, dtype=np.float)

        with open(endeffector_positions_file, 'r') as f:
            endeffector_positions = list(csv.reader(f, delimiter=","))
            endeffector_positions = np.array(endeffector_positions, dtype=np.float)

        return (action, endeffector_positions)
    
    def __getitem__(self, index):
        self.set_seed(index)
        if self.ordered:
            path = self.data_dirs[self.d]
            if self.d == len(self.data_dirs) - 1:
                self.d = 0
            else:
                self.d += 1
        else:
            path = self.data_dirs[np.random.randint(len(self.data_dirs))]

        cond =  self.get_csv(path)
        seq = self.get_seq(path)
        return seq, cond
