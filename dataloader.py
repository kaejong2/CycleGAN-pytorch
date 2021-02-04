
import os
import glob
import random
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, data_path, transforms_=None, mode='train'):
        self.transform = transforms_
        self.files_A = sorted(glob.glob(os.path.join(data_path,'%sA' % mode)+'/*'))
        self.files_B = sorted(glob.glob(os.path.join(data_path,'%sB' % mode)+'/*'))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)])
        img_B = Image.open(self.files_B[index % len(self.files_A)])
        # item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        # item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {'img_A': img_A, 'img_B': img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def data_loader(args, mode="train"):
    # Dataset loader
    transforms_ = transforms.Compose(
        [transforms.Resize((286,286), Image.BICUBIC), 
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageDataset(args.data_path, transforms_=transforms_, mode = mode)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    return dataloader


# args = Arguments()
# print(data_loader(args))
# data = data_loader(args)
# for i, data in enumerate(data):
#     print(data["A"])
#     print(data["B"])

