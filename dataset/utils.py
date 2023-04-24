import torch
import numpy as np
import h5py


# load and normalize data
def load_h5_data(args):
    num_points = args.num_points
    num_4X_points = int(args.num_points * 4)
    num_out_points = int(args.num_points * args.up_rate)
    skip_rate = args.skip_rate
    use_random_input = args.use_random_input
    h5_file_path = args.h5_file_path

    if use_random_input:
        with h5py.File(h5_file_path, 'r') as f:
            # (b, n, 3)
            input = f['poisson_%d' % num_4X_points][:]
            # (b, n, 3)
            gt = f['poisson_%d' % num_out_points][:]
    else:
        with h5py.File(h5_file_path, 'r') as f:
            input = f['poisson_%d' % num_points][:]
            gt = f['poisson_%d' % num_out_points][:]
    # (b, n, c)
    assert input.shape[0] == gt.shape[0]

    # (b, 1)
    data_radius = np.ones(shape=(input.shape[0], 1))
    # the center point of input
    input_centroid = np.mean(input, axis=1, keepdims=True)
    input = input - input_centroid
    # (b, 1)
    input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
    # normalize to a unit sphere
    input = input / np.expand_dims(input_furthest_distance, axis=-1)
    gt = gt - input_centroid
    gt = gt / np.expand_dims(input_furthest_distance, axis=-1)
    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]

    return input, gt, data_radius


# nonuniform sample point cloud to get input data
def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


# data augmentation
def jitter_perturbation_point_cloud(input, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = input.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += input
    return jittered_data


def rotate_point_cloud_and_gt(input, gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    input = np.dot(input, rotation_matrix)
    if gt is not None:
        gt = np.dot(gt, rotation_matrix)
    return input, gt


def random_scale_point_cloud_and_gt(input, gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(scale_low, scale_high)
    input = np.multiply(input, scale)
    if gt is not None:
        gt = np.multiply(gt, scale)
    return input, gt, scale