from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset
import numpy as np
import torchvision.transforms as transforms
import torch

class Dataset(TorchvisionDataset):

    def __init__(self, root: str):
        super().__init__(root)

        transform_train = transforms.Compose([transforms.CenterCrop(900), transforms.Resize(256), transforms.ToTensor()])
        transform_test = transforms.Compose([transforms.CenterCrop(900), transforms.Resize(256), transforms.ToTensor()])
        self.train_set = ImageFolder(root=self.root +'data/', transform =transform_train)
        self.test_set = ImageFolder(root=self.root + 'label/', transform=transform_test)
