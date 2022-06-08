import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import json
import os
from PIL import Image

def getData(mode, obj_dict):
    if mode == "train":
        with open("train.json", 'r') as f:
            j = json.loads(f.read())
        img_name = []
        label = []
        for key, val in j.items(): 
            img_name.append(key)
            label.append([obj_dict[item] for item in val])
        return img_name, label


def get_test_label(file):
    with open("objects.json", 'r') as f:
        objects = json.loads(f.read())
    with open(file, 'r') as f:
        test_list = json.loads(f.read())
    
    labels = torch.zeros(len(test_list), 24)
    for i in range(len(test_list)):
        for cond in test_list[i]:
            labels[i,int(objects[cond])] = 1
    return labels


class iclevrDataset(Dataset):
    def __init__(self, root = "./iclevr", mode = "train"):
        assert mode == "train"

        self.root = root
        self.mode = mode

        self.obj_dict = self.get_object_json()
        self.img_name, self.label = getData(mode, self.obj_dict)
        self.num_classes = 24

        self.transformations = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.img_name[index])).convert('RGB')
        img = self.transformations(img)

        condition = self.label[index]
        one_hot_condition = torch.zeros(self.num_classes)
        for i in condition:
            one_hot_condition[i] = 1.
        
        return img, one_hot_condition


    def get_object_json(self):
        with open("objects.json", 'r') as f:
            j = json.loads(f.read())
        return j



