import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

class DINOLossX(nn.Module):
    """ copied from https://github.com/facebookresearch/dino/blob/main/main_dino.py#L363
    adapted for X-modal loss calculation
    """
    def __init__(self, out_dim, 
                 center_momentum=0.9):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum

    def forward(self, student_output, teacher_output, 
                student_temp, teacher_temp, 
                num_student_views, num_teacher_views, 
                ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / student_temp
        student_out = student_out.chunk(num_student_views)

        # teacher centering and sharpening
        teacher_out = F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(num_teacher_views)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                # # commenting this as loss is calculated over cross-modal
                # if v == iq:
                #     # we skip cases where student and teacher operate on the same view
                #     continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        # dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        if dist.is_initialized(): # default case
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)             # local debug 

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class MMD_Loss(nn.Module):
    """ source: https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """
    def __init__(self, kernel_type = 'gaussian'):
        super().__init__()
        
        self.kernel_type = kernel_type
        
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K
    
    def gaussian_rbf_kernel(self, x, y, sigma_sqr=2., **kwargs):
        r"""
        Gaussian radial basis function (RBF) kernel.
        .. math::
            k(x, y) = \exp (\frac{||x-y||^2}{\sigma^2})
        """
        pairwise_distance_matrix = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1)
        K = torch.exp(-pairwise_distance_matrix / (1. * sigma_sqr))
        return K

    def forward(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy

        
        elif self.kernel_type == "mean_cov": # this is coral loss
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
        else:
            raise NotImplementedError()

        
        
        
        
        
        
        
        
        
        