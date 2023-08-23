#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import logging
import matplotlib
import os
import sys
import torch
# Fix problem: no $DISPLAY environment variable
matplotlib.use('Agg')

from argparse import ArgumentParser
from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net, batch_test


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Runner')
    parser.add_argument('--local_rank', dest='local_rank', type=int)
    parser.add_argument('--test', dest='test', help='Test neural networks', action='store_true')
    parser.add_argument('--batch_test', dest='batch_test',
                        help='Test neural networks for each view', action='store_true')
    parser.add_argument('--weights', dest='weights', help='Initialize network from the weights file', default=None)
    args = parser.parse_args()
    return args


def init_distributed_mode():
    torch.distributed.init_process_group(backend='nccl')
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(gpu)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('world size {}'.format(world_size))


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.weights is not None:
        cfg.CONST.WEIGHTS = args.weights
        if not args.test:
            cfg.TRAIN.RESUME_TRAIN = True

    # Set GPU and distributed data parallel to use
    init_distributed_mode()

    # Print config
    if torch.distributed.get_rank() == 0:
        print('Use config:')
        pprint(cfg)

    # Start train/test process
    if not args.test and not args.batch_test:
        train_net(cfg)
    else:
        if 'WEIGHTS' in cfg.CONST and os.path.exists(cfg.CONST.WEIGHTS):
            if args.test:
                test_net(cfg)
            elif args.batch_test:
                batch_test(cfg)
        else:
            logging.error('Please specify the file path of checkpoint.')
            sys.exit(2)


if __name__ == '__main__':
    if sys.version_info < (3, 0):
        raise Exception("Please follow the installation instruction on 'https://github.com/hzxie/Pix2Vox'")

    logging.basicConfig(format='[%(levelname)s] %(asctime)s %(message)s', level=logging.INFO)
    main()
