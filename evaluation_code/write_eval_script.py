import os
from glob import glob
import argparse


def write_eval_script(args):
    pcds = glob(os.path.join(args.upsampled_pcd_path, '*.xyz'))
    if args.dataset == 'pu1k':
        mesh_dir = "../data/PU1K/test/original_meshes/"
        script_name = "eval_pu1k.sh"
    else:
        mesh_dir = "../data/PU-GAN/test/"
        script_name = "eval_pugan.sh"
    with open(script_name, 'w') as f:
        for pcd_path in pcds:
            pcd_name = pcd_path.split("/")[-1]
            prefix = os.path.splitext(pcd_name)[0]
            prefix = prefix.split('_')[-1]
            if prefix == 'distance':
                continue
            mesh_name = pcd_name.replace(".xyz", ".off")
            mesh_path = os.path.join(mesh_dir, mesh_name)
            f.write("./evaluation {} {}\n".format(mesh_path, pcd_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation Arguments')
    parser.add_argument('--dataset', default='', type=str, help='datasetname, pu1k or pugan')
    parser.add_argument('--upsampled_pcd_path', default='', type=str, help='the upsampled point cloud path')
    args = parser.parse_args()

    assert args.upsampled_pcd_path != ''
    assert args.dataset in ['pu1k', 'pugan']

    write_eval_script(args)
