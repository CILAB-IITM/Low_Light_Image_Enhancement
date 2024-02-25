import torch.nn.functional as F
import torch
from models import register, make
import torch.nn as nn
from torch.nn import L1Loss

loss = L1Loss()

def model_loss_train_attn_only(disp_ests, disp_gt, mask):
    weights = [1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss_train_freeze_attn(disp_ests, disp_gt, mask):
    weights = [0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
    
def model_loss_train(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0] 
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)
    
def model_loss_test(disp_ests, disp_gt, mask):
    weights = [1.0] 
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)


#cfnet loss

@register('warping-loss')
class WarpLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, recons, pred, gt, inp, delta=1, weights = None):
        if weights is None:
            wts = [0.5 * 0.5, 0.5 * 0.7, 0.5 * 1.0, 1 * 0.5, 1 * 0.7, 1 * 1.0, 2 * 0.5, 2 * 0.7, 2 * 1.0]
        else:
            wts = weights
        left = inp[:,:int(inp.shape[1]/2),:,:]
        losses = []
        for recon, weight in zip(recons, wts):
            losses.append(weight*loss(recon, left))
        en_loss = loss(pred,gt)
        return sum(losses)+delta*en_loss

        


