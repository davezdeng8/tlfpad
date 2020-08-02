import argparse
import os
import signal
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from flownet3d import FlowNet3D
from losses import chamfer_distance, emd
from ecnds import ECNDS
from ecds import ECDS
from pnpp_models import PNPPNDS, PNPPDS

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
        return (pointcloud[0:4], pointcloud[3+self.steps])

def save_model(epoch, model, optimizer, scheduler):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    }, "checkpoint.tar")

def train():

    write_freq = 500

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    traindata = LoadData(args.train_dir, args.steps)
    data = DataLoader(traindata, batch_size = args.batch_size, shuffle = True, num_workers = 4)
    valdata = LoadData(args.val_dir, args.steps, length = 2500)
    datav = DataLoader(valdata, batch_size = 1, shuffle = True, num_workers = 0)

    write_step = int(len(traindata)/write_freq/args.batch_size)

    if args.model == 'ECNDS':
        model = ECNDS(args.knn_step, args.k)
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
        return
 
    opt = optim.AdamW(model.parameters(), lr=args.lr*100, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    if args.load_checkpoint!="":
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    model.to(device)

    def handler(sig, frame):
        save_model(i, model, opt, scheduler)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)

    writer = SummaryWriter()

    for i in range(args.epochs):
        if args.mode == "train":
            model.train()
        else:
            model.eval()
        write_loss = 0.0
        write_track_loss_cd = 0.0
        write_track_loss_emd = 0.0

        opt.zero_grad()
        for j, (x, ground_truth) in enumerate(data):
            x = x.to(device)
            ground_truth = ground_truth.to(device)
            for k in range(args.steps):
                pred = model(x)
                x = torch.cat((x[:,1:], torch.unsqueeze(pred, dim = 1)), dim = 1)

            cd_loss = chamfer_distance(ground_truth, pred)
            emd_loss = emd(ground_truth, pred)
            loss = args.cd_weight*cd_loss+args.emd_weight*emd_loss
            write_loss = write_loss+loss.item()
            write_track_loss_cd = write_track_loss_cd+cd_loss.item()
            write_track_loss_emd = write_track_loss_emd+emd_loss.item()
            loss = loss/args.grad_step_size
            loss.backward()
            print("training progress: " + str((j+1)/len(traindata)*args.batch_size), end="\r")

            if (j+1)%args.grad_step_size == 0:
                opt.step()
                scheduler.step()
                opt.zero_grad()

            if (j+1)%write_step == 0:
                writer.add_scalar("ptpredict/train_loss", write_loss/write_step, i*len(traindata)/args.batch_size+j+1)
                writer.add_scalar("ptpredict/chamfer_distance", write_track_loss_cd/write_step, i*len(traindata)/args.batch_size+j+1)
                writer.add_scalar("ptpredict/earth_movers_distance", write_track_loss_emd/write_step, i*len(traindata)/args.batch_size+j+1)
                write_loss = 0.0
                write_track_loss_cd = 0.0
                write_track_loss_emd = 0.0

        save_model(i, model, opt, scheduler)
        
        with torch.no_grad():
            accuracy_cd = 0.0
            accuracy_emd = 0.0
            countv = 0
            model.eval()
            for x, ground_truth in datav:
                x = x.to(device)
                ground_truth = ground_truth.to(device)
                for k in range(args.steps):
                    pred = model(x)
                    x = torch.cat((x[:,1:], torch.unsqueeze(pred, dim = 1)), dim = 1)
                cd_loss = chamfer_distance(ground_truth, pred)
                emd_loss = emd(ground_truth, pred)
                accuracy_cd = accuracy_cd + cd_loss
                accuracy_emd = accuracy_emd + emd_loss
                countv = countv+1
                print("Validation progress: " + str(countv/len(valdata)), end="\r")
            print("Accuracy (Chamfer Distance): " + str(accuracy_cd/countv))
            print("Accuracy (Earth Mover Distance): " + str(accuracy_emd/countv))
            writer.add_scalar("ptpredict/validation_cd", accuracy_cd/countv, i*len(valdata)+countv)
            writer.add_scalar("ptpredict/validation_emd", accuracy_emd/countv, i*len(valdata)+countv)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--lr', type = float, default = .00001, metavar='lr', help='learning rate')
    parser.add_argument('--load_checkpoint', type = str, default = "", help='location of model checkpoint; if training from scratch use empty string')
    parser.add_argument('--knn_step', type = int, default = 2200, help = 'points per batch of knn')
    parser.add_argument('--train_dir', type = str, default = "train_data")
    parser.add_argument('--val_dir', type = str, default = "validation_data")
    parser.add_argument('--grad_step_size', type = int, default = 8, help = 'number of batches before updating weights')
    parser.add_argument('--k', type = int, default = 16)
    parser.add_argument('--cd_weight', type = float, default = 1.0)
    parser.add_argument('--emd_weight', type = float, default = .02)
    parser.add_argument('--steps', type = int, default = 1, help = 'number of steps into the future')
    parser.add_argument('--mode', type = str, default = "train", help = 'training mode')
    parser.add_argument('--model', type = str, default = "ECNDS", help = 'which model to train')
    args = parser.parse_args()

    train()