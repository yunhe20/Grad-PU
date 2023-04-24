import argparse


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')