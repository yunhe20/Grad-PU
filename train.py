import os
import torch
import sys
sys.path.append(os.getcwd())
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import PUDataset
from models.P2PNet import P2PNet
from args.pu1k_args import parse_pu1k_args
from args.pugan_args import parse_pugan_args
from args.utils import str2bool
from models.utils import *
import argparse


def train(args):
    set_seed(args.seed)
    start = time.time()

    # load data
    train_dataset = PUDataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    # set up folders for checkpoints and logs
    str_time = datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger('train', log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    # create model
    logger.info('========== Build Model ==========')
    model = P2PNet(args)
    model = model.cuda()
    # get the parameter size
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))
    # log
    logger.info(args)
    logger.info(repr(model))
    # set model state
    model.train()

    # optimizer
    assert args.optim in ['adam', 'sgd']
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # lr scheduler
    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

    # train
    logger.info('========== Begin Training ==========')
    for epoch in range(args.epochs):
        logger.info('********* Epoch %d *********' % (epoch + 1))
        # epoch loss
        epoch_loss = 0.0

        for i, (input_pts, gt_pts, radius) in enumerate(train_loader):
            # (b, n, 3) -> (b, 3, n)
            input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()

            # # midpoint interpolation
            interpolate_pts = midpoint_interpolate(args, input_pts)

            # query points
            query_pts = get_query_points(interpolate_pts, args)
            # model forward, predict point-to-point distance: (b, 1, n)
            pred_p2p = model(interpolate_pts, query_pts)
            # calculate loss
            loss = get_p2p_loss(args, pred_p2p, query_pts, gt_pts)
            epoch_loss += loss.item()

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            writer.add_scalar('train/loss', loss, i)
            writer.flush()
            if (i+1) % args.print_rate == 0:
                logger.info("epoch: %d/%d, iters: %d/%d, loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss / (i+1)))

        # lr scheduler
        scheduler_steplr.step()

        # log
        interval = time.time() - start
        logger.info("epoch: %d/%d, avg epoch loss: %f, time: %d mins %.1f secs" %
          (epoch + 1, args.epochs, epoch_loss / len(train_loader), interval / 60, interval % 60))

        # save checkpoint
        if (epoch + 1) % args.save_rate == 0:
            model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
            model_path = os.path.join(ckpt_dir, model_name)
            torch.save(model.state_dict(), model_path)


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=60, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=10, type=int, help='model save frequency')
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['pu1k', 'pugan']

    if train_args.dataset == 'pu1k':
        model_args = parse_pu1k_args()
    else:
        model_args = parse_pugan_args()

    reset_model_args(train_args, model_args)

    train(model_args)
