import os
from glob import glob
import open3d as o3d
import numpy as np
import argparse
import torch
from tqdm import tqdm
from einops import rearrange
from models.utils import normalize_point_cloud, add_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PU-GAN Test Data Generation Arguments')
    parser.add_argument('--input_pts_num', default=2048, type=int, help='the input points number')
    parser.add_argument('--gt_pts_num', default=8192, type=int, help='the gt points number')
    parser.add_argument('--noise_level', default=0, type=float, help='the noise level')
    parser.add_argument('--jitter_max', default=0.03, type=float, help="jitter max")
    parser.add_argument('--mesh_dir', default='./data/PU-GAN/test/', type=str, help='input mesh dir')
    parser.add_argument('--save_dir', default='./data/PU-GAN/test_pointcloud/', type=str, help='output point cloud dir')
    args = parser.parse_args()

    dir_name = 'input_' + str(args.input_pts_num)
    if args.gt_pts_num % args.input_pts_num == 0:
        up_rate = args.gt_pts_num / args.input_pts_num
        dir_name += '_' + str(int(up_rate)) + 'X'
    else:
        up_rate = args.gt_pts_num / args.input_pts_num
        dir_name += '_' + str(up_rate) + 'X'
    if args.noise_level != 0:
        dir_name += '_noise_' + str(args.noise_level)
    input_save_dir = os.path.join(args.save_dir, dir_name, 'input_'+str(args.input_pts_num))
    if not os.path.exists(input_save_dir):
        os.makedirs(input_save_dir)
    gt_save_dir = os.path.join(args.save_dir, dir_name, 'gt_'+str(args.gt_pts_num))
    if not os.path.exists(gt_save_dir):
        os.makedirs(gt_save_dir)
    mesh_path = glob(os.path.join(args.mesh_dir, '*.off'))
    for i, path in tqdm(enumerate(mesh_path), desc='Processing'):
        pcd_name = path.split('/')[-1].replace(".off", ".xyz")
        mesh = o3d.io.read_triangle_mesh(path)
        # input pcd
        input_pcd = mesh.sample_points_poisson_disk(args.input_pts_num)
        input_pts = np.array(input_pcd.points)

        # add noise
        if args.noise_level != 0:
            input_pts = torch.from_numpy(input_pts).float().cuda()
            # (n, 3) -> (3, n)
            input_pts = rearrange(input_pts, 'n c -> c n').contiguous()
            # (3, n) -> (1, 3, n)
            input_pts = input_pts.unsqueeze(0)
            # normalize input
            input_pts, centroid, furthest_distance = normalize_point_cloud(input_pts)
            # add noise
            input_pts = add_noise(input_pts, sigma=args.noise_level, clamp=args.jitter_max)
            input_pts = centroid + input_pts * furthest_distance
            # (1, 3, n) -> (n, 3)
            input_pts = rearrange(input_pts.squeeze(0), 'c n -> n c').contiguous()
            input_pts = input_pts.detach().cpu().numpy()

        input_save_path = os.path.join(input_save_dir, pcd_name)
        np.savetxt(input_save_path, input_pts, fmt='%.6f')

        # gt pcd
        gt_pcd = mesh.sample_points_poisson_disk(args.gt_pts_num)
        gt_pts = np.array(gt_pcd.points)
        gt_save_path = os.path.join(gt_save_dir, pcd_name)
        np.savetxt(gt_save_path, gt_pts, fmt='%.6f')
