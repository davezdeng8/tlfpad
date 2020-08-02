from nuscenes.nuscenes import NuScenes
import torch
import numpy as np
import argparse
from distancebased_sort import distance_based_sort

def load(directory, version, batch_size):
    sweeps_per_scene = 400
    device = torch.device("cpu")
    nusc = NuScenes(version=version, dataroot=directory, verbose=False)

    for scene in nusc.scene:
        sample = nusc.get('sample', scene['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        for batch_per_scene in range(int(sweeps_per_scene/batch_size)):
            scene_out = torch.tensor([], device = device, dtype = torch.float)
            while sd_rec['next'] != '' and scene_out.shape[0]<batch_size:
                lidar_path = nusc.get_sample_data_path(sd_rec['token'])
                pointcloud = np.fromfile(lidar_path, dtype = np.float32).reshape((-1,5))[:,:3]
                sorted_pointcloud = distance_based_sort(pointcloud)[12000:34000]
                sweeps = torch.tensor(sorted_pointcloud, device = device, dtype = torch.float)
                shuffle = torch.randperm(sweeps.shape[0])
                sweeps = sweeps[shuffle]
                scene_out = torch.cat((scene_out, torch.unsqueeze(torch.t(sweeps), 0)), dim =0)
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            if scene_out.shape[0] == batch_size:
                yield scene_out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='save batches')
    parser.add_argument('--which', type = str, default = "train", help='train or test')
    parser.add_argument('--nuscenes_dir', type = str, default = "data/sets/nuscenes/", help='location of nuscenes dataset')
    args = parser.parse_args()

    if args.which == "train":
        train = load(args.nuscenes_dir, "v1.0-trainval", 8)
        directory = "train_data"
        for i, batch in enumerate(train):
            torch.save(batch, directory + "/training_batch_" + str(i) + ".pt")
            print(i, end="\r")
    elif args.which == "test":
        test = load(args.nuscenes_dir, "v1.0-test", 8)
        directory = "validation_data"
        for i, batch in enumerate(test):
            torch.save(batch, directory + "/test_batch_" + str(i) + ".pt")
            print(i, end="\r")
    else:
        print("put train or test")
