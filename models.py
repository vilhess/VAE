import torch
from torch import nn
import torch.nn.functional as F


class VariationalAutoEncoder_MNIST(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, h3_dim, h4_dim, h5_dim, z_dim):
        """
        function for our encoder and decoder

        Parameters
        ----------
        input_dim : int
                    the dimension of the input data
        h1_dim    : int
                    the dimension of our first hidden layer
        h2_dim    : int
                    the dimension of our second hidden layer
        h3_dim    : int
                    the dimension of our third hidden layer
        h4_dim    : int
                    the dimension of our fourth hidden layer
        h5_dim    : int
                    the dimension of our fifth hidden layer
        z_dim    : int
                    the dimension of our latent space
        """
        super().__init__()
        """function for our encoder and decoder"""
        # encoder
        self.img_2hid1 = nn.Linear(input_dim, h1_dim)
        self.hid1_2hid2 = nn.Linear(h1_dim, h2_dim)
        self.hid2_2hid3 = nn.Linear(h2_dim, h3_dim)
        self.hid3_2hid4 = nn.Linear(h3_dim, h4_dim)
        self.hid4_2hid5 = nn.Linear(h4_dim, h5_dim)
        self.hid2_2mu = nn.Linear(h5_dim, z_dim)
        self.hid2_2sigma = nn.Linear(h5_dim, z_dim)

        # decoder
        self.z_2hid5 = nn.Linear(z_dim, h5_dim)
        self.hid5_2hid4 = nn.Linear(h5_dim, h4_dim)
        self.hid4_2hid3 = nn.Linear(h4_dim, h3_dim)
        self.hid3_2hid2 = nn.Linear(h3_dim, h2_dim)
        self.hid2_2hid1 = nn.Linear(h2_dim, h1_dim)
        self.hid1_2img = nn.Linear(h1_dim, input_dim)

        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        """
        creation of our encoder simulating q_phi(z/x)

        Parameters
        ----------
        x : tensor

        Returns
        -------
        mu : tensor
        sigma : tensor
        """
        h1 = self.relu(self.img_2hid1(x))
        h2 = self.relu(self.hid1_2hid2(h1))
        h3 = self.relu(self.hid2_2hid3(h2))
        h4 = self.relu(self.hid3_2hid4(h3))
        h5 = self.relu(self.hid4_2hid5(h4))
        mu, sigma = self.hid2_2mu(h5), self.hid2_2sigma(h5)
        return mu, sigma

    def decoder(self, z):
        """
        creation of the decoder simulating p_theta(x/z)

        Parameters
        ----------
        z : tensor

        Returns
        -------
        the image generated
        """
        h5 = self.relu(self.z_2hid5(z))
        h4 = self.relu(self.hid5_2hid4(h5))
        h3 = self.relu(self.hid4_2hid3(h4))
        h2 = self.relu(self.hid3_2hid2(h3))
        h1 = self.relu(self.hid2_2hid1(h2))

        return torch.sigmoid(self.hid1_2img(h1))

    def forward(self, x):
        """Forward step with the reparameterization trick"""
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma


class VariationalAutoEncoder_FREY_FACE(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, h3_dim, z_dim):
        """
        function for our encoder and decoder

        Parameters
        ----------
        input_dim : int
                    the dimension of the input data
        h1_dim    : int
                    the dimension of our first hidden layer
        h2_dim    : int
                    the dimension of our second hidden layer
        h3_dim    : int
                    the dimension of our third hidden layer
        z_dim    : int
                    the dimension of our latent space
        """

        super().__init__()

        # encoder
        self.img_2hid1 = nn.Linear(input_dim, h1_dim)
        self.hid1_2hid2 = nn.Linear(h1_dim, h2_dim)
        self.hid2_2hid3 = nn.Linear(h2_dim, h3_dim)
        self.hid3_2mu = nn.Linear(h3_dim, z_dim)
        self.hid3_2sigma = nn.Linear(h3_dim, z_dim)

        # decoder
        self.z_2hid3 = nn.Linear(z_dim, h3_dim)
        self.hid3_2hid2 = nn.Linear(h3_dim, h2_dim)
        self.hid2_2hid1 = nn.Linear(h2_dim, h1_dim)
        self.hid1_2img = nn.Linear(h1_dim, input_dim)

        # activation function
        self.relu = nn.ReLU()

    def encode(self, x):
        """creation of our encoder simulating q_phi(z/x)

        Parameters
        ----------
        x : tensor

        Returns
        -------
        mu : tensor
        sigma : tensor
        """
        h1 = self.relu(self.img_2hid1(x))
        h2 = self.relu(self.hid1_2hid2(h1))
        h3 = self.relu(self.hid2_2hid3(h2))
        mu, sigma = self.hid3_2mu(h3), self.hid3_2sigma(h3)
        return mu, sigma

    def decoder(self, z):
        """
        creation of the decoder simulating p_theta(x/z)

        Parameters
        ----------
        z : tensor

        Returns
        -------
        the image generated
        """
        h3 = self.relu(self.z_2hid3(z))
        h2 = self.relu(self.hid3_2hid2(h3))
        h1 = self.relu(self.hid2_2hid1(h2))
        return torch.sigmoid(self.hid1_2img(h1))

    def forward(self, x):
        """Forward step with the reparameterization trick"""
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decoder(z_reparametrized)
        return x_reconstructed, mu, sigma


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(kernel_size=5, in_channels=1, out_channels=16)
        self.conv2 = nn.Conv2d(kernel_size=5, in_channels=16, out_channels=32)

        self.fc = nn.Linear(in_features=512, out_features=10)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
