import torch
import torch.nn as nn
from models.model_utils import EdgeConv, CBL, PointNet_FP_Module, FCFuseBlock, CAFFBlock, Fold, fps_sample_points
from utils.config import cfg_from_yaml_file
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformer import decTransformer


############################################################
# point transform
class pointTransform(nn.Module):
    def __init__(self, npoints, noise_dim=1, noise_stdv=2e-3, amplitude=1e-3):
        super(pointTransform, self).__init__()
        self.npoints = npoints
        self.noise_dim = noise_dim
        self.noise_stdv = noise_stdv
        self.amplitude = amplitude
        self.mlp = nn.Sequential(nn.Conv1d(4, 8, 1), nn.BatchNorm1d(8), nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(8, 6, 1), nn.BatchNorm1d(6), nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(6, 3, 1), nn.BatchNorm1d(3), nn.Tanh())

    def forward(self, points):
        device = points.device
        bs = points.shape[0]
        noise_points = torch.normal(mean=0, std=torch.ones((bs, self.noise_dim, self.npoints), device=device) * self.noise_stdv)
        noise_feature = torch.cat([points, noise_points], dim=1)
        relative_xyz = self.mlp(noise_feature)
        res_points = points + relative_xyz * self.amplitude

        return res_points


############################################################
# Edge extract feature modules
class EdgeExtractFeature(nn.Module):
    def __init__(self, stage_npoints):
        super(EdgeExtractFeature, self).__init__()
        self.edge1 = EdgeConv(input_channels=8, mlp=[64, 64, 128], out_npoints=stage_npoints[0])
        self.edge2 = EdgeConv(input_channels=128, mlp=[128, 128, 256], out_npoints=stage_npoints[1])
        self.edge3 = EdgeConv(input_channels=256, mlp=[256, 256, 512], out_npoints=stage_npoints[2])
        self.edge4 = EdgeConv(input_channels=512, mlp=[512, 512, 1024])

    def forward(self, in_features, points):
        stage1_features, stage1_points = self.edge1(in_features, points)
        stage2_features, stage2_points = self.edge2(stage1_features, stage1_points)
        stage3_features, stage3_points = self.edge3(stage2_features, stage2_points)

        global_feature = self.edge4(stage3_features, stage3_points)
        global_feature = global_feature.max(dim=-1, keepdim=False)[0]

        feature = (stage1_features, stage2_features, stage3_features)
        res_center = (stage1_points, stage2_points, stage3_points)

        return feature, res_center, global_feature


############################################################
# Feature Propagation-Transformer upsample block
class FP_UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_channels, fp_mlp):
        super(FP_UpsampleBlock, self).__init__()
        self.fp = PointNet_FP_Module(in_channel=in_channels, mlp=fp_mlp, use_points1=True,
                                     in_channel_points1=up_channels)
        # self.transformer = decTransformer(in_channels=fp_mlp[-1], out_channels=out_channels,
        #                                   guide_channels=global_channels)

    def forward(self, up_xyz, xyz, up_features, features):
        out_features = self.fp(up_xyz, xyz, up_features, features)
        # out_features = self.transformer(up_xyz, new_features, global_features)

        return out_features


############################################################
class HGGNet(nn.Module):
    def __init__(self, model_configs):
        super(HGGNet, self).__init__()
        global_channels = model_configs.global_channels
        # stage_npoints
        self.stage1_npoints = model_configs.stage1_npoints
        self.stage2_npoints = model_configs.stage2_npoints
        self.stage3_npoints = model_configs.stage3_npoints
        self.folding_step = model_configs.folding_step
        stage_npoints = (self.stage1_npoints, self.stage2_npoints, self.stage3_npoints)
        edge_channel = 8
        # self.pointTransform = pointTransform(2048)
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.edge_downsample = EdgeConv(input_channels=8, mlp=[8, edge_channel], out_npoints=1024)
        # self.repsurf = RepsurfExtractFeature(stage_npoints=stage_npoints, return_center=True, return_polar=True,
        #                                      cuda=True)
        self.edgeExtract = EdgeExtractFeature(stage_npoints)

        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=False),
            nn.Linear(1024, 3 * 1024)
        )

        self.dectrans = decTransformer(in_channels=512, out_channels=512, guide_channels=global_channels)

        self.conv1 = CBL(512, 512)
        self.up32 = FP_UpsampleBlock(in_channels=512, up_channels=256, fp_mlp=[256, 256])

        self.conv2 = CBL(256, 256)
        self.featureFuse1 = FCFuseBlock(channel=256)
        self.conv3 = CBL(256, 256)

        self.up21 = FP_UpsampleBlock(in_channels=256, up_channels=128, fp_mlp=[128, 128])
        self.conv4 = CBL(128, 128)
        self.featureFuse2 = FCFuseBlock(channel=128)

        self.conv5 = CBL(128, 128)
        # self.down12 = SurfaceAbstractionCD(npoint=self.stage2_npoints, radius=0.2, nsample=24, feat_channel=138,
        #                                    pos_channel=6, mlp=[128, 128, 256], group_all=False, return_polar=True,
        #                                    cuda=True)

        self.reduce_dim1 = nn.Linear(1027+128, 128)

        self.foldingNet = Fold(in_channel=128, step=self.folding_step, hidden_dim=128)  # rebuild a cluster point

        self.loss_func = ChamferDistanceL1()

    def forward(self, ptc):
        ptc = ptc.transpose(2, 1).contiguous()
        assert ptc.shape[1] == 3
        bs = ptc.shape[0]
        # ptc = self.pointTransform(ptc)
        input_feature = self.input_trans(ptc)
        edge_feature, ptc = self.edge_downsample(input_feature, ptc)
        stage_feature, stage_center, global_features = self.edgeExtract(edge_feature, ptc)
        stage1_feature, stage2_feature, stage3_feature = stage_feature
        stage1_center, stage2_center, stage3_center = stage_center

        decT_features = self.dectrans(stage3_center, stage3_feature, global_features)
        s3 = self.conv1(decT_features)
        s32 = self.up32(stage2_center, stage3_center, stage2_feature, s3)
        s2 = self.conv2(stage2_feature)
        s2_fuse1 = self.featureFuse1(s32, s2)
        s2_fuse1 = self.conv3(s2_fuse1)
        s21 = self.up21(stage1_center, stage2_center, stage1_feature, s2_fuse1)
        s1 = self.conv4(stage1_feature)
        s1_fuse = self.featureFuse2(s21, s1)
        s1_fuse = self.conv5(s1_fuse)

        head1 = s1_fuse.transpose(2, 1).contiguous()
        center_xyz = self.coarse_pred(global_features).reshape(bs, -1, 3)

        global_features = global_features.unsqueeze(1)

        # NOTE: foldingNet
        rebuild_feature = torch.cat([
            global_features.expand(-1, self.stage1_npoints, -1),
            head1,
            center_xyz], dim=2)  # B M 1027 + C
        rebuild_feature = self.reduce_dim1(rebuild_feature.reshape(bs * self.stage1_npoints, -1))  # BM C
        relative_xyz = self.foldingNet(rebuild_feature).reshape(bs, self.stage1_npoints, 3, -1)  # B M 3 S
        rebuild_points = (relative_xyz + center_xyz.unsqueeze(-1)).transpose(2, 3).reshape(bs, -1, 3)  # B N 3

        ret = (center_xyz, rebuild_points)

        return ret

    def get_loss(self, res, gt):
        sparse_loss = self.loss_func(res[0], gt)
        dense_loss = self.loss_func(res[1], gt)

        return sparse_loss, dense_loss


if __name__ == "__main__":
    device = torch.device("cuda")
    test_ptc = torch.rand(size=(4, 2048, 3), dtype=torch.float32).to(device)
    config = cfg_from_yaml_file("../cfgs/model_configs/HGGNet_one.yaml")
    test_module = HGGNet(config.model).to(device)
    dec = test_module(test_ptc)
    print("dec shape: ", dec[1].shape)

