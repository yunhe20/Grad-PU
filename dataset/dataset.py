import torch
import torch.utils.data as data
from dataset.utils import *


class PUDataset(data.Dataset):
    def __init__(self, args):
        super(PUDataset, self).__init__()

        self.args = args
        # input and gt: (b, n, 3) radius: (b, 1)
        self.input_data, self.gt_data, self.radius_data = load_h5_data(args)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        # (n, 3)
        input = self.input_data[index]
        gt = self.gt_data[index]
        radius = self.radius_data[index]
        if self.args.use_random_input:
            sample_idx = nonuniform_sampling(input.shape[0], sample_num=self.args.num_points)
            input = input[sample_idx, :]
        # data augmentation
        if self.args.use_random_input:
            input = jitter_perturbation_point_cloud(input, sigma=self.args.jitter_sigma, clip=self.args.jitter_max)
        input, gt = rotate_point_cloud_and_gt(input, gt)
        input, gt, scale = random_scale_point_cloud_and_gt(input, gt, scale_low=0.8, scale_high=1.2)
        radius = radius * scale
        # ndarray -> tensor
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)
        radius = torch.from_numpy(radius)
        return input, gt, radius