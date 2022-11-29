"""Global configuration"""
import torch
import os

# /
device = "cuda:0" if torch.cuda.is_available() else "cpu"
