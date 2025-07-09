import torch
import torch.nn as nn
from models.model_utils import fps_sample_points, knn_point, CBL, \
    LayerScale, Attention, DeformableLocalAttention, DeformableLocalCrossAttention, DynamicGraphAttention, \
    improvedDeformableLocalGraphAttention, Mlp, fps, CrossAttention
from utils.config import cfg_from_yaml_file
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from timm.models.layers import DropPath

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, \
        grouping_operation, three_interpolate, three_nn
except:
    raise Exception('Failed to load pointnet2_ops')


############################################################
# Edge extract feature modules
class EdgeSA(nn.Module):
    def __init__(self, in_channel, out_channel, k=16):
        super().__init__()
        '''
        K has to be 16
        '''
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(in_channel * 2, in_channel, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, in_channel),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        # self.layer2 = nn.Sequential(nn.Conv2d(in_channel * 2, out_channel, kernel_size=1, bias=False),
        #                             nn.GroupNorm(4, out_channel),
        #                             nn.LeakyReLU(negative_slope=0.2)
        #                             )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, 3, n
        fps_idx = furthest_point_sample(coor, num_group)  # b, n, 3

        combined_x = torch.cat([xyz, x], dim=1)

        new_combined_x = (
            gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_coor = new_coor.transpose(1, 2).contiguous()
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        # coor: bs, np, 3       x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k, coor_q)  # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, feature, pos, num):
        '''
            INPUT:
                feature : bs C N
                pos : bs N 3
                num : int 1024, 512
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128)
        '''
        coor, f_q = self.fps_downsample(pos, feature, num)
        f = self.get_graph_feature(coor, f_q, pos, feature)
        f = self.layer1(f)
        out_feature = f.max(dim=-1, keepdim=False)[0]

        # f = self.get_graph_feature(coor, f, coor, f)
        # f = self.layer2(f)
        # out_feature = f.max(dim=-1, keepdim=False)[0]

        return out_feature


class DGCNN_Grouper(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        '''
        K has to be 16
        '''
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 32),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 64),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        # self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
        #                             nn.GroupNorm(4, 64),
        #                             nn.LeakyReLU(negative_slope=0.2)
        #                             )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.layer5 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 128),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 256),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )
        self.layer7 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 256),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, 3, n
        fps_idx = furthest_point_sample(coor, num_group)

        combined_x = torch.cat([xyz, x], dim=1)

        new_combined_x = (
            gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        # coor: bs, np, 3     x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k, coor_q)  # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128)
        '''
        xyz = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(xyz)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        stage1_points, f_q = self.fps_downsample(coor, f, num[0])
        coor_d = stage1_points.transpose(2, 1).contiguous()
        f = self.get_graph_feature(coor_d, f_q, coor, f)
        f = self.layer2(f)
        stage1_feature = f.max(dim=-1, keepdim=False)[0]
        coor = coor_d

        # f = self.get_graph_feature(coor_d, f, coor_d, f)
        # f = self.layer3(f)
        # stage1_feature = f.max(dim=-1, keepdim=False)[0]
        # coor = coor_d

        stage2_points, f_q = self.fps_downsample(coor, stage1_feature, num[1])
        coor_d = stage2_points.transpose(2, 1).contiguous()
        f = self.get_graph_feature(coor_d, f_q, coor, stage1_feature)
        f = self.layer4(f)
        stage2_feature = f.max(dim=-1, keepdim=False)[0]
        coor = coor_d

        # f = self.get_graph_feature(coor_d, f, coor_d, f)
        # f = self.layer5(f)
        # stage2_feature = f.max(dim=-1, keepdim=False)[0]
        # coor = coor_d

        stage3_points, f_q = self.fps_downsample(coor, stage2_feature, num[2])
        coor_d = stage3_points.transpose(2, 1).contiguous()
        f = self.get_graph_feature(coor_d, f_q, coor, stage2_feature)
        f = self.layer6(f)
        f = f.max(dim=-1, keepdim=False)[0]

        f = self.get_graph_feature(coor_d, f, coor_d, f)
        f = self.layer7(f)
        stage3_feature = f.max(dim=-1, keepdim=False)[0]

        out_coor = stage3_points.transpose(2, 1).contiguous()
        out_f = stage3_feature.transpose(2, 1).contiguous()

        return out_coor, out_f


# Channels Attention Feature Fuse
class CAFFBlock(nn.Module):
    def __init__(self, channel, channel2, out_channel):
        super().__init__()
        in_channels = channel + channel2
        # cat feature channel --> channel weight
        self.CALayer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global average pooling: point feature --> channel feature
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels // 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels // 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid(),
        )

        self.reduce_dim = CBL(in_channels, out_channel)

    def forward(self, fea1, fea2):
        cat_features = torch.cat([fea1, fea2], dim=1)
        channel_weight = self.CALayer(cat_features)
        channel_attention = cat_features * channel_weight
        # res = self.reduce_dim(channel_attention)
        res = self.reduce_dim(channel_attention)

        return res


########## Transformer Encoder ##############################

class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'
            combine_style = 'onebyone'
    '''

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
    ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat',
                                 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph',
                                   'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                           attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'onebyone'
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'
            cross_attn_combine_style = 'concat'
    '''

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat',
                                           'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'

        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(
            self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph',
                                             'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                           proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                     attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                     n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat',
                                            'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'

        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(
            cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph',
                                              'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                                 proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                                      attn_drop=attn_drop, proj_drop=drop, k=k,
                                                                      n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx,
                                                           denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(
                    self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos,
                                                            idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(
                    self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos,
                                          idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos,
                                                        idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx)
        return x


class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """

    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
                 cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
                 k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i],
                cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx,
                      denoise_length=denoise_length)
        return q


############################################################
# Feature Propagation-Transformer upsample block

class PointNet_FP_Module(nn.Module):
    def __init__(self, in_channel, mlp, use_points1=False, in_channel_points1=None):
        """
        Args:
            in_channel: int, input channel of points2
            mlp: list of int
            use_points1: boolean, if use points
            in_channel_points1: int, input channel of points1
        """
        super(PointNet_FP_Module, self).__init__()
        self.use_points1 = use_points1

        if use_points1:
            in_channel += in_channel_points1

        last_channel = in_channel
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(CBL(last_channel, out_channel))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: Tensor, (B, N, 3)
            xyz2: Tensor, (B, M, 3)
            points1: Tensor, (B, p1_channels, N)
            points2: Tensor, (B, p2_channels, M)

        Returns:
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1, xyz2)
        dist = torch.clamp_min(dist, 1e-10)  # (B, N, 3)
        recip_dist = 1.0 / dist
        norm = torch.sum(recip_dist, 2, keepdim=True).repeat((1, 1, 3))
        weight = recip_dist / norm
        interpolated_points = three_interpolate(points2, idx, weight)  # B, in_channel, N

        if self.use_points1:
            new_points = torch.cat([interpolated_points, points1], 1)
        else:
            new_points = interpolated_points

        new_points = self.mlp_conv(new_points)
        return new_points


class decAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        q = q.transpose(2, 1)
        v = v.transpose(2, 1)
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(2, 1)
        return x


class decTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, guide_channels, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super(decTransformer, self).__init__()
        self.query_mlp = CBL(in_channels, out_channels)
        self.fuse_mlp = CBL(3 + guide_channels, out_channels)
        self.atten = decAttention(out_channels, out_channels, num_heads=num_heads, qkv_bias=qkv_bias,
                                  qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.out_mlp = CBL(out_channels, out_channels)

    def forward(self, in_pos, encode_features, global_features):
        in_pos = in_pos.transpose(2, 1)
        npoints = in_pos.shape[2]
        # generate v
        global_features = global_features.unsqueeze(-1).expand(-1, -1, npoints)
        # pos_features = self.pos_mlp(in_pos)
        v = torch.cat((in_pos, global_features), dim=1)

        v = self.fuse_mlp(v)
        # generate q
        q = self.query_mlp(encode_features)
        # attention
        trans_features = self.atten(q, v)
        res = self.out_mlp(q + trans_features)

        return res


class FPT_UpsampleBlock(nn.Module):
    def __init__(self, in_channels, fp_mlp, emb_dim):
        super(FPT_UpsampleBlock, self).__init__()
        self.fp = PointNet_FP_Module(in_channel=in_channels, mlp=fp_mlp, use_points1=False)
        self.proj = nn.Linear(fp_mlp[-1], emb_dim)
        self.mem_proj = nn.Linear(384, emb_dim)
        self.transformer = TransformerDecoder(
            embed_dim=emb_dim, depth=2, num_heads=4, k=8, n_group=2, mlp_ratio=2,
            self_attn_block_style_list=['attn-graph', 'attn'],
            self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-graph', 'attn'],
            cross_attn_combine_style='concat'
        )
        # self.transformer = decTransformer(in_channels=fp_mlp[-1], out_channels=out_channels,
        #                                   guide_channels=global_channels)

    def forward(self, up_xyz, xyz, features, mem, xyz1):
        new_features = self.fp(up_xyz, xyz, None, features)
        new_features = new_features.transpose(1, 2)
        new_features = self.proj(new_features)
        v = self.mem_proj(mem)
        out_features = self.transformer(q=new_features, v=v, q_pos=up_xyz, v_pos=xyz1, denoise_length=None)

        return out_features


########### rebuild #########################################
class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.step = step
        self.layer = Mlp(input_dims * 2, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature

        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


class CABlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        in_channels = channel
        # cat feature channel --> channel weight
        self.CALayer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # global average pooling: point feature --> channel feature
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid(),
        )

        self.reduce_dim = CBL(in_channels, channel)

    def forward(self, fea1):
        channel_weight = self.CALayer(fea1)
        channel_attention = fea1 * channel_weight
        # res = self.reduce_dim(channel_attention)
        res = self.reduce_dim(channel_attention)

        return res


############################################################
class HGGNet(nn.Module):
    def __init__(self, model_configs):
        super(HGGNet, self).__init__()
        global_channels = model_configs.global_channels
        # stage_npoints
        stage1_npoints = model_configs.stage1_npoints
        stage2_npoints = model_configs.stage2_npoints
        stage3_npoints = model_configs.stage3_npoints
        self.stage_npoints = (stage1_npoints, stage2_npoints, stage3_npoints)

        edge_channel = 8
        # self.pointTransform = pointTransform(2048)
        # self.input_trans = nn.Conv1d(3, 8, 1)
        # self.edge_downsample = EdgeConv(input_channels=8, mlp=[8, edge_channel], out_npoints=1024)
        # self.repsurf = RepsurfExtractFeature(stage_npoints=stage_npoints, return_center=True, return_polar=True,
        #                                      cuda=True)
        # self.edgeExtract = EdgeExtractFeature(stage_npoints)

        self.dgcnn = DGCNN_Grouper()

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 384)
        )

        self.input_proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 384)
        )

        self.self_encoder = TransformerEncoder(
            embed_dim=384, depth=4, num_heads=6, k=8, mlp_ratio=2, combine_style='concat',
            block_style_list=['attn-graph', 'attn', 'attn', 'attn'])

        self.increase_dim = nn.Sequential(
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024))

        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * 512)
        )

        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.up32 = FPT_UpsampleBlock(in_channels=384, fp_mlp=[384, 384], emb_dim=256)
        # self.featureFuse1 = CABlock(channel=256)

        self.up21 = FPT_UpsampleBlock(in_channels=256, fp_mlp=[256, 256], emb_dim=256)
        # self.featureFuse2 = CABlock(channel=256)
        # self.proj_dim = nn.Linear(384, 256)

        self.down12 = EdgeSA(in_channel=256, out_channel=256, k=16)
        self.featureFuse3 = CAFFBlock(channel=256, channel2=256, out_channel=256)

        self.down23 = EdgeSA(in_channel=256, out_channel=256, k=16)
        self.featureFuse4 = CAFFBlock(channel=256, channel2=384, out_channel=256)

        self.reduce_dim1 = nn.Linear(1027 + 256, 384)
        self.reduce_dim2 = nn.Linear(1027 + 256, 384)
        self.reduce_dim3 = nn.Linear(1027 + 256, 384)

        self.rebuild1 = SimpleRebuildFCLayer(384, step=32, hidden_dim=512)
        self.rebuild2 = SimpleRebuildFCLayer(384, step=32, hidden_dim=512)
        self.rebuild3 = SimpleRebuildFCLayer(384, step=32, hidden_dim=512)

        # self.foldingNet1 = Fold(in_channel=128, step=self.folding_step, hidden_dim=128)  # rebuild a cluster point
        # self.foldingNet2 = Fold(in_channel=128, step=self.folding_step, hidden_dim=128)
        # self.foldingNet3 = Fold(in_channel=128, step=self.folding_step, hidden_dim=128)

        self.loss_func = ChamferDistanceL1()

    # def get_loss(self, ret, gt, epoch=1):
    #     pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret

    #     assert pred_fine.size(1) == gt.size(1)

    #     # denoise loss
    #     idx = knn_point(self.factor, gt, denoised_coarse) # B n k
    #     denoised_target = index_points(gt, idx) # B n k 3
    #     denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
    #     assert denoised_target.size(1) == denoised_fine.size(1)
    #     loss_denoised = self.loss_func(denoised_fine, denoised_target)
    #     loss_denoised = loss_denoised * 0.5

    #     # recon loss
    #     loss_coarse = self.loss_func(pred_coarse, gt)
    #     loss_fine = self.loss_func(pred_fine, gt)
    #     loss_recon = loss_coarse + loss_fine

    #     return loss_denoised, loss_recon

    def forward(self, ptc):
        # ptc = ptc.transpose(2, 1).contiguous()
        assert ptc.shape[2] == 3
        bs = ptc.shape[0]
        # ptc = self.pointTransform(ptc)
        stage_center, stage_feature = self.dgcnn(ptc, self.stage_npoints)

        pe = self.pos_embed(stage_center)
        x = self.input_proj(stage_feature)

        x = self.self_encoder(x + pe, stage_center)

        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)
        coarse_inp = fps(ptc, 256)  # B 256 3
        coarse = torch.cat([coarse, coarse_inp], dim=1)  # B 768 3

        # query selection
        query_ranking = self.query_ranking(coarse)  # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True)  # b n 1
        coarse = torch.gather(coarse, 1, idx[:, :512].expand(-1, -1, coarse.size(-1)))

        coarse_3 = coarse[:, :128, :].contiguous()
        coarse_2 = coarse[:, 128:288, :].contiguous()
        coarse_1 = coarse[:, 288:, :].contiguous()

        feature = x.transpose(1, 2).contiguous()

        s2 = self.up32(coarse_2, coarse_3, feature, mem=x, xyz1=coarse_3)  # e
        s2 = s2.transpose(1, 2).contiguous()

        s1 = self.up21(coarse_1, coarse_2, s2, mem=x, xyz1=coarse_3)
        s1_t = s1.transpose(1, 2).contiguous()

        s12_down = self.down12(s1_t, coarse_1, num=160)
        s2_fuse = self.featureFuse3(s12_down, s2)

        s23_down = self.down23(s2_fuse, coarse_2, num=128)
        s3_fuse = self.featureFuse4(s23_down, feature)

        head1 = s1
        head2 = s2_fuse.transpose(2, 1).contiguous()
        head3 = s3_fuse.transpose(2, 1).contiguous()

        global_features = global_feature.unsqueeze(1)

        # NOTE: foldingNet
        rebuild3_feature = torch.cat([
            global_features.expand(-1, 128, -1),
            head3,
            coarse_3], dim=2)  # B M 1027 + C
        rebuild3_feature = self.reduce_dim3(rebuild3_feature)  # B M C
        relative3_xyz = self.rebuild3(rebuild3_feature).reshape(bs, 128, 3, -1)  # B M 3 S
        rebuild3_points = (relative3_xyz + coarse_3.unsqueeze(-1)).transpose(2, 3).reshape(bs, -1, 3)  # B N 3

        rebuild2_feature = torch.cat([
            global_features.expand(-1, 160, -1),
            head2,
            coarse_2], dim=2)  # B M 1027 + C
        rebuild2_feature = self.reduce_dim2(rebuild2_feature)  # B M C
        relative2_xyz = self.rebuild2(rebuild2_feature).reshape(bs, 160, 3, -1)  # B M 3 S
        rebuild2_points = (relative2_xyz + coarse_2.unsqueeze(-1)).transpose(2, 3).reshape(bs, -1, 3)  # B N 3

        rebuild1_feature = torch.cat([
            global_features.expand(-1, 224, -1),
            head1,
            coarse_1], dim=2)  # B M 1027 + C
        rebuild1_feature = self.reduce_dim1(rebuild1_feature)  # B M C
        relative1_xyz = self.rebuild1(rebuild1_feature).reshape(bs, 224, 3, -1)  # B M 3 S
        rebuild1_points = (relative1_xyz + coarse_1.unsqueeze(-1)).transpose(2, 3).reshape(bs, -1, 3)  # B N 3

        # cat the input
        rebuild_points = torch.cat([rebuild1_points, rebuild2_points, rebuild3_points], dim=1).contiguous()

        ret = (coarse, rebuild_points)

        return ret

    def get_loss(self, res, gt):
        sparse_loss = self.loss_func(res[0], gt)
        dense_loss = self.loss_func(res[1], gt)

        return sparse_loss, dense_loss


if __name__ == "__main__":
    device = torch.device("cuda")
    test_ptc = torch.rand(size=(4, 2048, 3), dtype=torch.float32).to(device)
    test_fea = torch.rand(size=(4, 8, 2048), dtype=torch.float32).to(device)
    config = cfg_from_yaml_file("../cfgs/model_configs/HGGNet_6_26.yaml")
    test_module = HGGNet(config.model).to(device)
    dec = test_module(test_ptc)
    print("dec shape: ", dec[1].shape)

