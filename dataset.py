import torch
from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import albumentations as A
import torchvision
class FlickerDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path,transform = None,perc=1.0):
        #Naively, Load all caption data in memory
        assert 0.0 < perc <= 1.0
        self.perc = perc
        self.folder_path = folder_path
        self.caption_df = pd.read_csv(os.path.join(self.folder_path,'captions3.txt')).dropna(axis=0).drop_duplicates(subset="image")
        #Default transform handling
        if transform == None :
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return int(len(self.caption_df)*self.perc)

    def __getitem__(self, idx):

        imgname,caption,type_ = self.caption_df.iloc[idx,:]
        caption = caption
        img = Image.open(os.path.join(self.folder_path,'Images',imgname))
        #img = np.asarray(img)
        img = self.transform(img)

        return torch.Tensor(img), caption
    
    def filter_df(self, name):
        self.caption_df = self.caption_df[self.caption_df['type'] == name]
        return self


class COCODataset(torch.utils.data.Dataset):
    def __init__(self,img_folder,anon_path,transform,perc =1.0):
        assert 0.0 < perc <= 1.0
        self.perc = perc
        self.ds = torchvision.datasets.CocoCaptions(root=img_folder,annFile=anon_path,transform=transform)
    
    def __len__(self):
        return int(self.perc*len(self.ds))
    
    def __getitem__(self,idx):
        img,caption = self.ds[idx]
        return img, caption[0]

transform_train = \
    transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])


transform_test = \
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.50707537, 0.48654878, 0.44091785),
            (0.267337, 0.2564412, 0.27615348)),
    ])


class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self,root,train,transform):
        self.ds = torchvision.datasets.CIFAR100(root=root,train=train,download=True,transform=transform)
        self.class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        self.prompt = "A photo of "

    def __len__(self):
        return int(len(self.ds))

    def __getitem__(self,idx):
        #import pdb; pdb.set_trace()
        data = self.ds[idx]
        img = data[0]
        #import pdb;pdb.set_trace()
        txt = self.prompt + self.class_names[data[1]]
        return img,txt



