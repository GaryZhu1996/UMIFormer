#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Developed by Zhenwei Zhu <garyzhu1996@gmail.com>

import warnings
import torch
from einops import rearrange

from models.encoder.encoder_vit import default_cfgs, vit_cfg, VisionTransformer, DistilledVisionTransformer


class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        model_name = cfg.NETWORK.ENCODER.VIT_IVDB.MODEL_NAME
        self.decoupling_type = instantiate(
            inheritance=DistilledVisionTransformer if 'distilled' in model_name else VisionTransformer,
            type_id=cfg.NETWORK.ENCODER.VIT_IVDB.TYPE)
        self.encoder = self.create_model(
            model_name=model_name,
            pretrained=cfg.NETWORK.ENCODER.VIT_IVDB.PRETRAINED,
            use_cls_token=cfg.NETWORK.ENCODER.VIT_IVDB.USE_CLS_TOKEN,
            block_types_list=cfg.NETWORK.ENCODER.VIT_IVDB.BLOCK_TYPES_LIST,
            k=cfg.NETWORK.ENCODER.VIT_IVDB.K,
        )

    def _create_vision_transformer(self, variant, pretrained=False, **kwargs):
        default_cfg = default_cfgs[variant]
        default_num_classes = default_cfg['num_classes']
        default_img_size = default_cfg['input_size'][-1]
    
        num_classes = kwargs.pop('num_classes', default_num_classes)
        img_size = kwargs.pop('img_size', default_img_size)
        repr_size = kwargs.pop('representation_size', None)
        if repr_size is not None and num_classes != default_num_classes:
            # Remove representation layer if fine-tuning. This may not always be the desired action,
            # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
            warnings.warn("Removing representation layer for fine-tuning.")
            repr_size = None

        model = self.decoupling_type(
            img_size=img_size, num_classes=num_classes, representation_size=repr_size, **kwargs)
        model.default_cfg = default_cfg  # ??

        if pretrained:
            model.load_pretrained(variant)

        return model

    def create_model(self, model_name, pretrained=True, **kwargs):
        if model_name in vit_cfg.keys():
            model_kwargs = {**vit_cfg[model_name], **kwargs}
            if model_name == 'vit_small_patch16_224' and pretrained:
                model_kwargs.setdefault('qk_scale', 768 ** -0.5)
            return self._create_vision_transformer(
                model_name, pretrained=pretrained, **model_kwargs)
        else:
            raise ValueError('Unsupported model')

    def forward(self, images):
        # x [B, V, C, H, W]
        batch_size, view_num = images.shape[0:2]
        feature = self.encoder(images)  # [B*V, P, D]
        feature = rearrange(feature, '(b v) l d -> b v l d', b=batch_size, v=view_num)  # [B, V, P, D]
        return feature


def instantiate(inheritance=VisionTransformer, type_id=0):
    # super class
    class VITDecouplingBase(inheritance):
        def __init__(self,
                     block_types_list,  # 0 for intra-image processing block; 1 for inter-image processing block
                     k,
                     *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.block_types_list = block_types_list
            
            # add inter-image processing blocks on specified locations
            if self.block_types_list.count(0) != len(self.blocks):
                raise ValueError(
                    'Number of intra-image processing block does not match to the vision transformer depth.')

            # edgeconv
            self.k = k
            self.fusion_layers = torch.nn.ModuleList(
                [torch.nn.Linear(self.embed_dim, self.embed_dim, bias=False)] * self.block_types_list.count(1))
            self.fusion_layers.apply(self._init_weights)

        def grouping(self, x):
            # x [B, V, P, D]
            with torch.no_grad():
                batch_size, view_num, patch_num, dim = x.shape

                x = rearrange(x, 'b v n d -> b (v n) d')
                sum_sq_x = torch.sum(x ** 2, dim=-1)  # [B, V*P]
                sum_x = torch.einsum('b i d, b j d -> b i j', x, x)  # [B, V*P, V*P]
                distances = torch.sqrt(sum_sq_x.unsqueeze(1) + sum_sq_x.unsqueeze(2) - 2 * sum_x)
                distances = rearrange(
                    distances, 'b (v n) (v1 n1) -> b v n v1 n1', v=view_num, v1=view_num)  # [B, V, P, V, P]
        
                x = x.unsqueeze(dim=-2).expand(-1, -1, view_num * patch_num, -1)  # [B, V, P, V, P, D]
                x = rearrange(x, 'b (v n) (v1 n1) d -> b v1 n1 v n d', v=view_num, v1=view_num)
        
                _, indices = torch.sort(distances, dim=-1)
                _, indices = torch.sort(indices, dim=-1)
                index = torch.where(indices < self.k)  # [B*V*P*V*K]
                neighbor = rearrange(x[index], '(b v n v1 k) d -> b v n v1 k d',
                                     b=batch_size, v=view_num, n=patch_num, v1=view_num, k=self.k)

                triu = torch.triu(torch.ones(view_num, view_num), diagonal=1).cuda()
                triu = triu[None, :, None, :, None, None].expand(batch_size, -1, patch_num, -1, self.k, dim)
                tril = torch.tril(torch.ones(view_num, view_num), diagonal=-1).cuda()
                tril = tril[None, :, None, :, None, None].expand(batch_size, -1, patch_num, -1, self.k, dim)
                neighbor = (neighbor * triu)[:, :, :, 1:, ::] + (neighbor * tril)[:, :, :, :-1, ::]
                neighbor = rearrange(neighbor, 'b v n v1 k d -> b v n (v1 k) d')

            return neighbor
        
        def weighted_sum(self, x, count):
            # [B, V, P, K, D]
            weights = self.fusion_layers[count](x)
            weights = torch.softmax(weights, dim=3)
            x = x * weights
            x = torch.sum(x, dim=3)  # [B, V, P, D]
            return x

        def forward(self, x):
            # x [B, V, C, H, W]
            view_num = x.shape[1]
            x, batch_size = self.prepare(x)
            # [B*V, P, D]
    
            intra_count = 0
            inter_count = 0
            for i in range(len(self.block_types_list)):
                if self.block_types_list[i] == 0:
                    x = self.blocks[intra_count](x)
                    # [B*V, P, D]
                    intra_count += 1
                else:
                    x = rearrange(x, '(b v) n d -> b v n d', v=view_num)
                    # [B, V, P, D]
                    neighbor = self.grouping(x)  # [B, V, P, (V-1)*k, D]
                    x = self.similar_correlation(x, neighbor, view_num, inter_count)  # [B, V, P, D]
                    x = rearrange(x, 'b v n d -> (b v) n d')
                    inter_count += 1
    
            x = self.norm(x)
            x = self.pre_logits(x)
            return x


    # VIT IVDB - Offset (2*dim -> dim)
    # token_feature = token_feature + feature_offset(token_feature, fusion_method(edge_features))
    # edge_feature_ij = edge_cal_function(x_j-x_i)
    class VIT_IVDB_Offset(VITDecouplingBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if torch.distributed.get_rank() == 0:
                print('Encoder: VIT_Decoupling_Similar_Correlation - Offset (2*dim -> dim)')

            self.edge_fc = torch.nn.ModuleList(
                [torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, self.embed_dim),
                    torch.nn.GELU())] * self.block_types_list.count(1))

            self.feature_offset = torch.nn.ModuleList(
                [torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim * 2, self.embed_dim))
                ] * self.block_types_list.count(1))

            self.edge_fc.apply(self._init_weights)
            self.feature_offset.apply(self._init_weights)

        def similar_correlation(self, token_feature, neighbor, view_num, count):
            # token_features [B, V, P, D]; neighbor [B, V, P, (V-1)*k, D]
            edge_features = self.edge_fc[count](
                neighbor - token_feature.unsqueeze(dim=3).expand(-1, -1, -1, (view_num - 1) * self.k, -1))
            # [B, V, P, (V-1)*k, D]

            edge_features = self.weighted_sum(edge_features, count)
            # [B, V, P, D]

            token_feature = \
                token_feature + self.feature_offset[count](torch.cat((token_feature, edge_features), dim=-1))

            return token_feature  # [B, V, P, D]


    # VIT IVDB - Offset Weight (2*dim -> dim)
    # token_feature = token_feature * (1 + feature_offset_weight(token_feature, fusion_method(edge_features))
    # edge_feature_ij = edge_cal_function(x_j-x_i)
    class VIT_IVDB_Offset_Weight(VITDecouplingBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
            if torch.distributed.get_rank() == 0:
                print('Encoder: VIT_Decoupling_Similar_Correlation - Offset Weight (2*dim -> dim)')
    
            self.edge_fc = torch.nn.ModuleList(
                [torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim, self.embed_dim),
                    torch.nn.GELU())] * self.block_types_list.count(1))
    
            self.feature_offset_weight = torch.nn.ModuleList(
                [torch.nn.Sequential(
                    torch.nn.Linear(self.embed_dim * 2, self.embed_dim),
                    torch.nn.Tanh())] * self.block_types_list.count(1))
    
            self.edge_fc.apply(self._init_weights)
            self.feature_offset_weight.apply(self._init_weights)

        def similar_correlation(self, token_feature, neighbor, view_num, count):
            # token_feature [B, V, P, D]; neighbor [B, V, P, (V-1)*k, D]
            edge_features = self.edge_fc[count](
                neighbor - token_feature.unsqueeze(dim=3).expand(-1, -1, -1, (view_num - 1) * self.k, -1))
            # [B, V, P, (V-1)*k, D]
    
            edge_features = self.weighted_sum(edge_features, count)
            # [B, V, P, D]

            token_feature = token_feature * (
                    1 + self.feature_offset_weight[count](torch.cat((token_feature, edge_features), dim=-1)))

            return token_feature  # [B, V, P, D]

    decoupling_types = [VIT_IVDB_Offset, VIT_IVDB_Offset_Weight]
    return decoupling_types[type_id]
