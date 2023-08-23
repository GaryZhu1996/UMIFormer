#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import torch
from einops import rearrange
from timm.models.vision_transformer import partial

from models.transformer_base.decoder.standard_layers import Block


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        
        if torch.distributed.get_rank() == 0:
            print('Decoder: 3D-RETR')

        if cfg.NETWORK.DECODER.VOXEL_SIZE % 4 != 0:
            raise ValueError('voxel_size must be dividable by patch_num')

        # default
        num_resnet_blocks = 2
        num_cnn_layers = 3
        cnn_hidden_dim = 64

        self.patch_num = 4 ** 3
        self.trans_patch_size = 4
        self.voxel_size = cfg.NETWORK.DECODER.VOXEL_SIZE
        self.patch_size = cfg.NETWORK.DECODER.VOXEL_SIZE // self.patch_num

        self.transformer_decoder = TransformerDecoder(
            embed_dim=cfg.NETWORK.DECODER.RETR.DIM,
            num_heads=cfg.NETWORK.DECODER.RETR.HEADS,
            depth=cfg.NETWORK.DECODER.RETR.DEPTH,)
            # attn_drop=0.4)

        self.layer_norm = torch.nn.LayerNorm(cfg.NETWORK.DECODER.RETR.DIM)

        has_resblocks = num_resnet_blocks > 0
        dec_chans = [cnn_hidden_dim] * num_cnn_layers
        dec_init_chan = cfg.NETWORK.DECODER.RETR.DIM if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]
        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        dec_layers = []
        for (dec_in, dec_out) in dec_chans_io:
            dec_layers.append(
                torch.nn.Sequential(
                    torch.nn.ConvTranspose3d(
                        dec_in, dec_out, 4, stride=2, padding=1),
                    torch.nn.ReLU()))
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
        if num_resnet_blocks > 0:
            dec_layers.insert(0, torch.nn.Conv3d(cfg.NETWORK.DECODER.RETR.DIM, dec_chans[1], 1))
        dec_layers.append(torch.nn.Conv3d(dec_chans[-1], 1, 1))
        self.decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, context):
        # [B, P, D]
        out = self.transformer_decoder(context=context)  # [B, P, D]
        out = self.layer_norm(out)
        out = rearrange(out, 'b (h w c) d -> b d h w c',
                        h=self.trans_patch_size,
                        w=self.trans_patch_size,
                        c=self.trans_patch_size)  # [B, D, H, W, C]
        out = self.decoder(out)
        return torch.sigmoid(out)


class ResBlock(torch.nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(c, c, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(c, c, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(c, c, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class TransformerDecoder(torch.nn.Module):
    def __init__(
            self,
            patch_num=4 ** 3,
            embed_dim=768,
            num_heads=12,
            depth=8,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_num = patch_num
        norm_layer = norm_layer or partial(torch.nn.LayerNorm)  # eps=1e-6  ???
        self.emb = torch.nn.Embedding(patch_num, embed_dim)
        self.blocks = torch.nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth)])
    
    def forward(self, context):
        x = self.emb(torch.arange(self.patch_num, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        for blk in self.blocks:
            x = blk(x=x, context=context)
        return x
