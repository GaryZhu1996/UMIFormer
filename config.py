#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

from easydict import EasyDict as edict

__C = edict()
cfg = __C


# Dataset Config
__C.DATASETS = edict()
__C.DATASETS.SHAPENET = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH = '/home/zzw/datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH = '/home/zzw/datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'


# Dataset
__C.DATASET = edict()
__C.DATASET.TRAIN_DATASET = 'ShapeNet'  # 'ShapeNetChairRFC'
__C.DATASET.TEST_DATASET = 'ShapeNet'  # 'Pix3D'


# Common
__C.CONST = edict()
__C.CONST.RNG_SEED = 0
__C.CONST.IMG_W = 224  # Image width for input
__C.CONST.IMG_H = 224  # Image height for input
__C.CONST.CROP_IMG_W = 128  # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H = 128  # Dummy property for Pascal 3D
__C.CONST.BATCH_SIZE_PER_GPU = 16  # 16  # for train only
__C.CONST.N_VIEWS_RENDERING = 3
__C.CONST.NUM_WORKER = 20  # number of data workers
__C.CONST.WEIGHTS = './pths/UMIFormer.pth'


# Directories
__C.DIR = edict()
__C.DIR.OUT_PATH = './output/'


# Network
__C.NETWORK = edict()
__C.NETWORK.ENCODER = edict()

# vit
__C.NETWORK.ENCODER.VIT_IVDB = edict()
__C.NETWORK.ENCODER.VIT_IVDB.MODEL_NAME = 'vit_deit_base_distilled_patch16_224'
__C.NETWORK.ENCODER.VIT_IVDB.PRETRAINED = True
__C.NETWORK.ENCODER.VIT_IVDB.USE_CLS_TOKEN = False
# vit-ivdb
__C.NETWORK.ENCODER.VIT_IVDB = edict()
__C.NETWORK.ENCODER.VIT_IVDB.MODEL_NAME = 'vit_deit_base_distilled_patch16_224'
__C.NETWORK.ENCODER.VIT_IVDB.PRETRAINED = True
__C.NETWORK.ENCODER.VIT_IVDB.USE_CLS_TOKEN = False
__C.NETWORK.ENCODER.VIT_IVDB.BLOCK_TYPES_LIST = [0, 0, 0, 1] * 4
__C.NETWORK.ENCODER.VIT_IVDB.TYPE = 1  # 0 for offset(2*dim->dim); 1 for offset weight(2dim->dim)
__C.NETWORK.ENCODER.VIT_IVDB.K = 5


__C.NETWORK.DECODER = edict()
__C.NETWORK.DECODER.VOXEL_SIZE = 32
# retr
__C.NETWORK.DECODER.RETR = edict()
__C.NETWORK.DECODER.RETR.DEPTH = 8
__C.NETWORK.DECODER.RETR.HEADS = 12
__C.NETWORK.DECODER.RETR.DIM = 768


__C.NETWORK.MERGER = edict()
__C.NETWORK.MERGER.WITHOUT_PARAMETERS = False
# stm
__C.NETWORK.MERGER.STM = edict()
__C.NETWORK.MERGER.STM.DIM = 768
__C.NETWORK.MERGER.STM.OUT_TOKEN_LENS = [196, 196]
__C.NETWORK.MERGER.STM.K = 15
__C.NETWORK.MERGER.STM.NUM_HEAD = 12


# Training
__C.TRAIN = edict()
__C.TRAIN.RESUME_TRAIN = False
__C.TRAIN.SYNC_BN = True
__C.TRAIN.NUM_EPOCHS = 150
__C.TRAIN.BRIGHTNESS = .4
__C.TRAIN.CONTRAST = .4
__C.TRAIN.SATURATION = .4
__C.TRAIN.NOISE_STD = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE = [[225, 255], [225, 255], [225, 255]]

__C.TRAIN.ENCODER_LEARNING_RATE = 1e-4
__C.TRAIN.DECODER_LEARNING_RATE = 1e-4
__C.TRAIN.MERGER_LEARNING_RATE = 1e-4

__C.TRAIN.LR_scheduler = 'MilestonesLR'  # 'ExponentialLR' or 'MilestonesLR'
__C.TRAIN.WARMUP = 0
# for ExponentialLR
__C.TRAIN.EXPONENTIALLR = edict()
__C.TRAIN.EXPONENTIALLR.SCHEDULE_FACTOR = 1
# for MilestonesLR
__C.TRAIN.MILESTONESLR = edict()
__C.TRAIN.MILESTONESLR.ENCODER_LR_MILESTONES = [50, 120]
__C.TRAIN.MILESTONESLR.DECODER_LR_MILESTONES = [50, 120]
__C.TRAIN.MILESTONESLR.MERGER_LR_MILESTONES = [50, 120]
__C.TRAIN.MILESTONESLR.GAMMA = .1

__C.TRAIN.BETAS = (.9, .999)
__C.TRAIN.SAVE_FREQ = 10  # weights will be overwritten every save_freq epoch
__C.TRAIN.SHOW_TRAIN_STATE = 500

__C.TRAIN.LOSS = 2  # 1 for 'bce'; 2 for 'dice'; 3 for 'ce_dice'; 4 for 'focal'

__C.TRAIN.TEST_AFTER_TRAIN = True


# Testing options
__C.TEST = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH = [.3, .4, .5, .6]
