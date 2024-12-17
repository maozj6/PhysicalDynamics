import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader
# from loader5 import VaeDataset,SequenceDataset
from tqdm import tqdm
import torch.nn.functional as F
# Define the VAE architecture
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import numpy as np
# from skimage.metrics import structural_similarity as ssim
import random
# from nltk.corpus import words
# import nltk

class Dynamicst():
    def __init__(self, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, gravity=9.8, tau=0.02):
        """
        Initialize the parameters for the Dynamics class.

        Parameters:
        - gravity: Gravitational acceleration (default 9.8 m/s^2)
        - masscart: Mass of the cart (default 1.0 kg)
        - masspole: Mass of the pole (default 0.1 kg)
        - length: Half the length of the pole (default 0.5 m)
        - force_mag: Magnitude of the force applied (default 10.0 N)
        - tau: Time step for updates (default 0.02 s)
        """
        self.gravity = gravity
        self.masscart = masscart
        self.masspole = masspole
        self.total_mass = masscart + masspole
        self.length = length  # half-pole length
        self.polemass_length = masspole * length
        self.force_mag = force_mag
        self.tau = tau

    def forward(self, z, action):
        # Extract state variables
        x, x_dot, theta, theta_dot = z[ 0], z[ 1], z[2], z[ 3]

        # Convert action to force
        force = self.force_mag * (action * 2 - 1)  # action == 1 -> 10.0, action == 0 -> -10.0

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Update values
        x_updated = x + self.tau * x_dot
        x_dot_updated = x_dot + self.tau * xacc
        theta_updated = theta + self.tau * theta_dot
        theta_dot_updated = theta_dot + self.tau * thetaacc

        # Create a new z tensor to avoid in-place modification
        z_updated = z.copy()
        z_updated[0] = x_updated
        z_updated[ 1] = x_dot_updated
        z_updated[2] = theta_updated
        z_updated[3] = theta_dot_updated
        return z_updated

# 定义自定义的线性层
class CustomLinearLayer(nn.Module):


    def __init__(self):
        super(CustomLinearLayer, self).__init__()
        # 初始化权重和偏置


        self.masspole = nn.Parameter(torch.randn(1))

        self.length =  nn.Parameter(torch.randn(1))

        self.force_mag =  nn.Parameter(torch.randn(1))
        self.masscart =  nn.Parameter(torch.randn(1))

        self.tau=0.02
        self.gravity = 9.8

    def forward(self, z,action):
        weight_masspole =  0.01 + (1 - 0.01) * torch.sigmoid(self.masspole)

        weight_length =  0.05 + (5 - 0.05) * torch.sigmoid(self.length)

        weight_masscart =  0.1 + (10 - 0.1) * torch.sigmoid(self.masscart )
        weight_force_mag =  1 + (100 - 1) * torch.sigmoid(self.force_mag)

        # 应用线性方程
        # 从z中提取x, x_dot, theta, theta_dot
        x, x_dot, theta, theta_dot = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        x = x/0.625
        theta = theta*60*torch.pi/180
        # Convert action to force
        force = weight_force_mag * (action * 2 - 1)  # action == 1 -> 10.0, action == 0 -> -10.0

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (force + (weight_masspole * weight_length) * theta_dot.pow(2) * sintheta) / (weight_masspole + weight_masscart)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                weight_length * (4.0 / 3.0 -weight_masspole * costheta.pow(2) / (weight_masspole + weight_masscart)))
        xacc = temp - (weight_masspole * weight_length) * thetaacc * costheta / (weight_masspole + weight_masscart)

        # 更新值，避免使用原地操作
        x_updated = x + self.tau * x_dot
        x_dot_updated = x_dot + self.tau * xacc
        theta_updated = theta + self.tau * theta_dot
        theta_dot_updated = theta_dot + self.tau * thetaacc
        x_updated = x_updated * 0.625
        theta_updated = theta_updated / 60 / torch.pi * 180
        # 创建一个新的z张量以避免原地修改
        z_updated = torch.clone(z)
        z_updated[:, 0] = x_updated
        z_updated[:, 1] = x_dot_updated
        z_updated[:, 2] = theta_updated
        z_updated[:, 3] = theta_dot_updated
        return z_updated

class DVAE(nn.Module):
    def __init__(self, latent_size=4):
        super(DVAE, self).__init__()
        self.latent_size = latent_size

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)  # Input: (3, 96, 96) -> Output: (32, 48, 48)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (32, 48, 48) -> (64, 24, 24)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (64, 24, 24) -> (128, 12, 12)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (128, 12, 12) -> (256, 6, 6)
        self.fc_mu = nn.Linear(256 * 6 * 6, latent_size)
        self.fc_logvar = nn.Linear(256 * 6 * 6, latent_size)

        # Decoder
        self.dec_fc = nn.Linear(latent_size, 256 * 6 * 6)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.dec_conv5 = nn.ConvTranspose2d(16, 8, kernel_size=6, stride=2, padding=2)
        # self.dec_conv6 = nn.ConvTranspose2d(8, 3, kernel_size=6, stride=1, padding=2)
        self.dynamics = CustomLinearLayer()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(-1, 256, 6, 6)  # Reshape to match the beginning shape of the decoder
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x = F.relu(self.dec_conv4(x))
        # x = F.relu(self.dec_conv5(x))
        # x = torch.sigmoid(self.dec_conv6(x))  # Ensure the output is in [0, 1]
        return x


if __name__ == '__main__':
    device = torch.device("cpu")
    model = DVAE(latent_size=8).to(device)
    para = torch.load("models/1action5_vibraphone_check_point_model.pth",map_location=torch.device('cpu'))
    model.load_state_dict(para['model_state_dict'])
    randomTensor = torch.tensor([-1.5,0,0,0,0,0,0,0],dtype=float)
    decoded = model.decode(randomTensor.float().view(1,8))
    image_tensor = decoded.squeeze(0)# [3, 96, 96]
    # 将 tensor 转为 NumPy 格式，并调整维度顺序 (C, H, W) -> (H, W, C)
    image_np = image_tensor.permute(1, 2, 0).detach().numpy()

    # 显示图像
    plt.imshow(image_np)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    print()
