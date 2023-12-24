import random
import torch
from datasets.videotransforms.utils import functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation

    Given mean: m and std: s
    will  normalize each channel as channel = (channel - mean) / std

    Args:
        mean (int): mean value
        std (int): std value
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of stacked images or image
            of size (C, H, W) to be normalized

        Returns:
            Tensor: Normalized stack of image of image
        """
        return F.normalize(tensor, self.mean, self.std)


class SpatialRandomCrop(object):
    """Crops a random spatial crop in a spatio-temporal
    numpy or tensor input [Channel, Time, Height, Width]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple): in format (height, width)
        """
        self.size = size

    def __call__(self, tensor):
        h, w = self.size
        _, _, tensor_h, tensor_w = tensor.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)
        cropped = tensor[:, :, y1:y1 + h, x1:x1 + h]
        return cropped


# class MinMaxNormalize(object):
#     """Normalize a tensor in between minimum and maximum

#     X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#     X_scaled = X_std * (max - min) + min

#     Args:
#         minimum (int): mean value
#         maximum (int): std value
#     """

#     def __init__(self, minimum=-1, maximum=1):
#         self.minimum = minimum
#         self.maximum = maximum

#     def __call__(self, tensor):
#         """
#         Args:
#             tensor (Tensor): Tensor of stacked images or image
#             of size (C, H, W) to be normalized

#         Returns:
#             Tensor: Normalized stack of image of image
#         """
#         # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#         # tensor_std = (tensor - torch.min(tensor, dim=0)[0]) / (torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0])
#         max_val = [tensor[ch, ::].max() for ch in range(tensor.shape[0])]
#         min_val = [tensor[ch, ::].min() for ch in range(tensor.shape[0])]
#         max_val = torch.tensor(max_val).view(3, 1, 1, 1)
#         min_val = torch.tensor(min_val).view(3, 1, 1, 1)
        
#         tensor_transformed = (tensor - min_val) / (max_val - min_val)
        
        
        
        
#         return F.normalize(tensor, self.mean, self.std)
