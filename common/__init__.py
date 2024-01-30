import torch
from torch.utils.data import Dataset
from PIL import Image


def prepare_imgpaths_prompts(source):
    """
    Args:
        source -- images dir
    Return:
        imgpaths -- list of image paths
        prompts -- list of image prompts
    """
    pass


def load_images(imgpaths):
    """
    Args:
        imgpaths -- list of image paths
    Return:
        images --  images tensor, dtype uint8, shape (n,c,h,w)
    """
    pass
