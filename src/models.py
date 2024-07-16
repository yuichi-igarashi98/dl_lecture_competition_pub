import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 150
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Softmax(dim = 1), 
            nn.Linear(hid_dim, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.45,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size = 10, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size = 5, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size = 3, padding="same")

        self.pool0 = nn.MaxPool1d(kernel_size = 10, stride = 2)
        self.pool1 = nn.MaxPool1d(kernel_size = 5, stride = 2)
        self.pool2 = nn.MaxPool1d(kernel_size = 3, stride = 2)

        
        self.batchnorm0 = nn.BatchNorm1d(num_features = out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features = out_dim)
        self.batchnorm2 = nn.BatchNorm1d(num_features = out_dim)

        self.relu0 =         nn.ReLU()
        self.relu1 =         nn.ReLU()
        self.relu2 =         nn.ReLU()

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
      if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
      else:
            X = self.conv0(X) 
      X = self.batchnorm0(X)
      X = self.relu0(X)
      X = self.pool0(X)

      X = self.conv1(X)
      X = self.batchnorm1(X)
      X = self.relu1(X)
      X = self.pool1(X)

      X = self.conv2(X)
      X = self.batchnorm2(X)
      X = self.relu2(X)
      X = self.pool2(X)
        
      return self.dropout(X)