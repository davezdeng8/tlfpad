import torch
import torch.nn as nn
from util_pnpp import PointNetSetAbstraction, PointNetFeaturePropogation, FlowEmbedding, PointNetSetUpConv


class PNPPNDS(nn.Module):
    def __init__(self):
        super(PNPPNDS, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=22000, radius=0.7, nsample=16, in_channel=5, mlp=[32,32], group_all=False)
        self.fe1 = FlowEmbedding(radius=1.0, nsample=24, in_channel = 32, mlp=[32], pooling='max', corr_func='concat')

        self.sa2 = PointNetSetAbstraction(npoint=22000, radius=0.7, nsample=16, in_channel=32, mlp=[64, 64], group_all=False)
        self.fe2 = FlowEmbedding(radius=1.0, nsample=24, in_channel = 64, mlp=[64], pooling='max', corr_func='concat')

        self.sa3 = PointNetSetAbstraction(npoint=22000, radius=0.7, nsample=16, in_channel=64, mlp=[128], group_all=False)

        self.head = nn.Sequential(nn.Conv1d(224, 512, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(512),
                                    nn.Conv1d(512, 256, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(256),
                                    nn.Conv1d(256, 128, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(128),
                                    nn.Conv1d(128, 3, kernel_size = 1))

    def forward(self, x):
        B, T, C, N = x.shape
        
        l1a_pc0, l1a_ft0 = self.sa1(x[:,0].contiguous(), torch.zeros(B, 5, N).to('cuda:0'))
        l1a_pc1, l1a_ft1 = self.sa1(x[:,1].contiguous(), torch.zeros(B, 5, N).to('cuda:0'))
        l1a_pc2, l1a_ft2 = self.sa1(x[:,2].contiguous(), torch.zeros(B, 5, N).to('cuda:0'))
        l1a_pc3, l1a_ft3 = self.sa1(x[:,3].contiguous(), torch.zeros(B, 5, N).to('cuda:0'))

        l1b_pc0, l1b_ft0 = self.fe1(l1a_pc1, l1a_pc0, l1a_ft1, l1a_ft0)
        l1b_pc1, l1b_ft1 = self.fe1(l1a_pc3, l1a_pc2, l1a_ft3, l1a_ft2)

        l2a_pc0, l2a_ft0 = self.sa2(l1b_pc0, l1b_ft0)
        l2a_pc1, l2a_ft1 = self.sa2(l1b_pc1, l1b_ft1)

        l2b_pc, l2b_ft = self.fe2(l2a_pc1, l2a_pc0, l2a_ft1, l2a_ft0)       

        l3_pc, l3_ft = self.sa3(l2b_pc, l2b_ft)

        aggr = torch.cat((l1a_ft3, l2a_ft1, l3_ft), dim = 1)

        flow = self.head(aggr)

        return x[:,3]+flow


class PNPPDS(nn.Module):
    def __init__(self):
        super(PNPPDS, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=5500, radius=0.5, nsample=16, in_channel=1, mlp=[128,128], group_all=False)
        self.fe1 = FlowEmbedding(radius=1.5, nsample=24, in_channel = 128, mlp=[128], pooling='max', corr_func='concat')

        self.sa2 = PointNetSetAbstraction(npoint=1375, radius=1, nsample=16, in_channel=128, mlp=[256, 256], group_all=False)
        self.fe2 = FlowEmbedding(radius=3, nsample=24, in_channel = 256, mlp=[256], pooling='max', corr_func='concat')

        self.sa3 = PointNetSetAbstraction(npoint=275, radius=2, nsample=16, in_channel=256, mlp=[512], group_all=False)

        self.su1 = PointNetSetUpConv(nsample=16, radius=4, f1_channel = 256, f2_channel = 512, mlp=[512], mlp2=[512])
        self.su2 = PointNetSetUpConv(nsample=16, radius=2, f1_channel = 128, f2_channel = 512, mlp=[512], mlp2=[512])
        self.fp = PointNetFeaturePropogation(in_channel = 512+1, mlp = [256])

        self.head = nn.Sequential(nn.Conv1d(256, 128, kernel_size = 1), nn.LeakyReLU(negative_slope=.2), nn.BatchNorm1d(128),
                                    nn.Conv1d(128, 3, kernel_size = 1))


    def forward(self, x):
        B, T, C, N = x.shape
        
        l1a_pc0, l1a_ft0 = self.sa1(x[:,0].contiguous(), torch.zeros(B, 1, N).to('cuda:0'))
        l1a_pc1, l1a_ft1 = self.sa1(x[:,1].contiguous(), torch.zeros(B, 1, N).to('cuda:0'))
        l1a_pc2, l1a_ft2 = self.sa1(x[:,2].contiguous(), torch.zeros(B, 1, N).to('cuda:0'))
        l1a_pc3, l1a_ft3 = self.sa1(x[:,3].contiguous(), torch.zeros(B, 1, N).to('cuda:0'))

        l1b_pc0, l1b_ft0 = self.fe1(l1a_pc1, l1a_pc0, l1a_ft1, l1a_ft0)
        l1b_pc1, l1b_ft1 = self.fe1(l1a_pc3, l1a_pc2, l1a_ft3, l1a_ft2)

        l2a_pc0, l2a_ft0 = self.sa2(l1b_pc0, l1b_ft0)
        l2a_pc1, l2a_ft1 = self.sa2(l1b_pc1, l1b_ft1)

        l2b_pc, l2b_ft = self.fe2(l2a_pc1, l2a_pc0, l2a_ft1, l2a_ft0)       

        l3_pc, l3_ft = self.sa3(l2b_pc, l2b_ft)

        l2_fnew1 = self.su1(l2a_pc1, l3_pc, l2a_ft1, l3_ft)
        l1_fnew1 = self.su2(l1a_pc3, l2a_pc1, l1a_ft3, l2_fnew1)
        l0_fnew1 = self.fp(x[:,3].contiguous(), l1a_pc3, torch.zeros(B, 1, N).to('cuda:0'), l1_fnew1)

        flow = self.head(l0_fnew1)

        return x[:,3]+flow

