#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def knn(x, xp, k):
    inner = -2*torch.matmul(x.transpose(2, 1), xp)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    xpxp = torch.sum(xp**2, dim = 1, keepdim=True)
    pairwise_distance = -xpxp - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def knn_mem(x, xp, k, step):
    device = x.device
    output = torch.tensor([], device = device, dtype = torch.long)
    for i in range(0, x.shape[-1], step):
        if i+step<x.shape[-1]:
            output = torch.cat((output, knn(x[:,:,i:i+step], xp, k)), dim = 1)
        else:
            output = torch.cat((output, knn(x[:,:,i:x.shape[-1]], xp, k)), dim = 1)
    return output

def get_graph_feature(x, xp, step, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_mem(x, xp, k, step)   # (batch_size, num_points, k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    xp = xp.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = xp.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    return feature.contiguous()


class EdgeConv(nn.Module):
    def __init__(self, features_in, features_out, k, step):
        super(EdgeConv, self).__init__()
        self.k = k
        self.bn = nn.BatchNorm2d(features_out)
        self.conv = nn.Sequential(nn.Conv2d(2*features_in, features_out, kernel_size = 1), nn.LeakyReLU(negative_slope=0.2), self.bn)
        self.step = step

    def forward(self, x, xp = None):
        if xp is None:
            xp = x
        x = get_graph_feature(x,xp, self.step, k=self.k)
        x = self.conv(x)
        return x.max(dim=-1, keepdim=False)[0]

class EdgeConvCustom(nn.Module):
    def __init__(self, network, k, step):
        super(EdgeConvCustom, self).__init__()
        self.k = k
        self.conv = network
        self.step = step

    def forward(self, x, xp = None):
        if xp is None:
            xp = x
        x = get_graph_feature(x,xp, self.step, k=self.k)
        x = self.conv(x)
        return x.max(dim=-1, keepdim=False)[0]

class ECNDS(nn.Module):
    # takes in (batch size, time steps (4), xyz, n points), outputs (batch size, xyz, n points)
    def __init__(self, step, k):
        super(ECNDS, self).__init__()
        #
        self.k = k
        self.flowscale = 1.5

        self.customlayer1 = nn.Sequential(nn.Conv2d(2*3, 32, kernel_size = 1), nn.LeakyReLU(negative_slope=0.2), nn.BatchNorm2d(32),
                                            nn.Conv2d(32, 32, kernel_size = 1), nn.LeakyReLU(negative_slope=0.2), nn.BatchNorm2d(32))
        self.customlayer2 = nn.Sequential(nn.Conv2d(2*32, 64, kernel_size = 1), nn.LeakyReLU(negative_slope=0.2), nn.BatchNorm2d(64),
                                            nn.Conv2d(64, 64, kernel_size = 1), nn.LeakyReLU(negative_slope=0.2), nn.BatchNorm2d(64))

        self.edgeconv1a = EdgeConvCustom(self.customlayer1, self.k, step)
        self.edgeconv1b = EdgeConv(32, 32, int(self.flowscale*self.k), step)
        self.edgeconv2a = EdgeConvCustom(self.customlayer2, self.k, step)
        self.edgeconv2b = EdgeConv(64, 64, int(self.flowscale*self.k), step)
        self.edgeconv3 = EdgeConv(64, 128, self.k, step)
        self.head = nn.Sequential(nn.Conv1d(224, 512, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(512),
                                    nn.Conv1d(512, 256, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(256),
                                    nn.Conv1d(256, 128, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(128),
                                    nn.Conv1d(128, 3, kernel_size = 1))

    def forward(self, x):
        x_layer1a = []
        for xi in x.transpose(0,1):
            x_layer1a.append(torch.unsqueeze(self.edgeconv1a(xi), dim = 0))
        x_layer1a = torch.cat(x_layer1a, dim = 0)

        x_layer1b_1 = self.edgeconv1b(x_layer1a[1], x_layer1a[0])
        x_layer1b_2 = self.edgeconv1b(x_layer1a[3], x_layer1a[2])

        x_layer2a_1 = self.edgeconv2a(x_layer1b_1)
        x_layer2a_2 = self.edgeconv2a(x_layer1b_2)

        x_layer2b = self.edgeconv2b(x_layer2a_2, x_layer2a_1)

        x_layer3 = self.edgeconv3(x_layer2b)

        x_aggr = torch.cat((x_layer1a[3], x_layer2a_2, x_layer3), dim = 1)

        flow = self.head(x_aggr)

        return x.transpose(0,1)[3]+flow

        
if __name__ == "__main__":
    points = torch.tensor([[[0,0,0],[0,0,1],[3,3,3],[3,3,2]]])
    points = points.transpose(2,1).to("cuda:0" if torch.cuda.is_available() else "cpu").float()
    points2 = points+1
    print(get_graph_feature(points,points2, k=2))
