import torch
from chamfer_distance import ChamferDistance
from PyTorchEMD.emd import earth_mover_distance

chamfer_dist = ChamferDistance()

def one_sided_chamfer_distance(pc1, pc2):
    loss = chamfer_dist(pc1.transpose(1,2), pc2.transpose(1,2))
    return torch.mean(loss[0])

def chamfer_distance(pc1, pc2):
    loss = chamfer_dist(pc1.transpose(1,2), pc2.transpose(1,2))
    return torch.mean(loss[0]+loss[1])/2

def emd(pc1, pc2):
    return torch.mean(earth_mover_distance(pc1, pc2)/pc1.shape[2])