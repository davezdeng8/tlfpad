import torch
import numpy as np
from losses import chamfer_distance_b, emd_b
from ecnds import ECNDS
from pnpp_models import PNPPDS, PNPPNDS
from ecds import ECDS
from flownet3d import FlowNet3D
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import time

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--checkpoint', type=str, default="models/ecnds.tar")
parser.add_argument("--steps", type=int, default=5)
parser.add_argument("--batch_size", type=int, default = 4)
parser.add_argument("--model", type=str, default = "ECNDS")
args = parser.parse_args()

class LoadData(Dataset):
    def __init__(self, directory, steps, length = -1):
        super(LoadData, self).__init__()
        self.dir = directory
        self.pointclouds = [os.path.join(self.dir, x) for x in os.listdir(self.dir)]
        if length == -1:
            self.length = len(self.pointclouds)
        else:
            self.length = length
        self.pointclouds = self.pointclouds[:self.length]
        self.steps = steps

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        pointcloud = torch.load(self.pointclouds[index])
        return (pointcloud[0:4], pointcloud[4:4+self.steps])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(args.checkpoint, map_location = device)

if args.model == 'ECNDS':
    model = ECNDS(2200, 16)
elif args.model == 'ECDS':
    model = ECDS()
elif args.model == 'PNPPNDS':
    model = PNPPNDS()
elif args.model == 'PNPPDS':
    model = PNPPDS()
elif args.model == 'FlowNet3D':
    model = FlowNet3D()
else:
    print("not a valid model")

model.load_state_dict(checkpoint['model_state_dict'])
print("model loaded")
model = model.to(device)
model.eval()

val_dir = "test_data"
valdata = LoadData(val_dir, args.steps, length = -1)
datav = DataLoader(valdata, batch_size = args.batch_size, shuffle = False, num_workers = 0)

history_cd2 = [[] for _ in range(args.steps)]
history_emd2 = [[] for _ in range(args.steps)]
# history_time = []
# history_mem = []
with torch.no_grad():
    for i, (x, ground_truth) in enumerate(datav):
        x = x.to(device)
        ground_truth = ground_truth.to(device)
        for k in range(args.steps):
            # torch.cuda.reset_max_memory_allocated()
            # start = time.time()
            pred = model(x)
            # history_time.append(time.time()-start)
            # history_mem.append(torch.cuda.max_memory_allocated())
            x = torch.cat((x[:,1:], torch.unsqueeze(pred, dim = 1)), dim = 1)
            gt = ground_truth[:, k]

            cd2 = chamfer_distance_b(pred, gt)
            emd2 = emd_b(pred, gt)
            history_cd2[k].extend(cd2)
            history_emd2[k].extend(emd2)

        print(i, end="\r")
        if i%200 == 0:
            for k in range(args.steps):
                print(np.mean(history_cd2[k]))
                print(np.mean(history_emd2[k]))
            # print(np.mean(history_time))
            # print(np.mean(history_mem))
            print()

    np.save("our_history_cd.npy", np.array(history_cd2))
    np.save("our_history_emd.npy", np.array(history_emd2))

    for i in range(args.steps):
        print(np.mean(history_cd2[i]))
        print(np.mean(history_emd2[i]))