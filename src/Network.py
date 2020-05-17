import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from base.base_net import BaseNet

class Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv0 = nn.Conv2d(3, 32, 8, stride=2, padding=3, bias=True)
        self.bn0 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv1 = nn.Conv2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)

        # Representation Layer

        # Decoder
        self.deconv0 = nn.ConvTranspose2d(32, 3, 8, stride=2, padding=3, bias=True)
        self.debn0 = nn.BatchNorm2d(3, eps=1e-04, affine=False)

        self.deconv1 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1, bias=True)
        self.debn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)



    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu(self.bn0(x))
        x = self.conv1(x)
        x = F.leaky_relu(self.bn1(x))
        # x = self.conv2(x)
        # x = F.leaky_relu(self.bn2(x))
        # x = self.conv3(x)
        # x = F.leaky_relu(self.bn3(x))
        # x = self.conv4(x)
        # x = F.leaky_relu(self.bn4(x))
        # x = self.conv5(x)
        # x = F.leaky_relu(self.bn5(x))
        # x = self.conv6(x)
        # x = F.leaky_relu(self.bn6(x))
        # x = self.conv7(x)
        # x = F.leaky_relu(self.bn7(x))
        # x = self.conv8(x)
        # x = F.leaky_relu(self.bn8(x))
        # x = self.conv9(x)
        # x = F.leaky_relu(self.bn9(x))
        #
        #
        # x = self.deconv9(x)
        # x = F.leaky_relu(self.debn9(x))
        # x = self.deconv8(x)
        # x = F.leaky_relu(self.debn8(x))
        # x = self.deconv7(x)
        # x = F.leaky_relu(self.debn7(x))
        # x = self.deconv6(x)
        # x = F.leaky_relu(self.debn6(x))
        # x = self.deconv5(x)
        # x = F.leaky_relu(self.debn5(x))
        # x = self.deconv4(x)
        # x = F.leaky_relu(self.debn4(x))
        # x = self.deconv3(x)
        # x = F.leaky_relu(self.debn3(x))
        # x = self.deconv2(x)
        # x = F.leaky_relu(self.debn2(x))
        x = self.deconv1(x)
        x = F.leaky_relu(self.debn1(x))
        x = self.deconv0(x)
        #x = F.leaky_relu(self.debn0(x))
        x = torch.sigmoid(x)

        return  x

