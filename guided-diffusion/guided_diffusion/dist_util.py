"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# guided_diffusion/dist_util.py
import torch


def setup_dist():
    # No-op; we’re running single‐process
    pass

def dev():
    # Returns GPU if available, else CPU
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_rank():
    return _rank

def get_world_size():
    return _world_size
