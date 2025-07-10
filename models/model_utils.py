# from sklearn.utils import resample
import torch
import torch.nn as nn
import einops

try:
    from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, ball_query, \
        grouping_operation, three_interpolate, three_nn
except:
    raise Exception('Failed to load pointnet2_ops')

# from knn_cuda import KNN
k=16
# repsurfKnn = KNN(k=13, transpose_mode=False)
# edgeKnn = KNN(k=k, transpose_mode=False)


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist   


class Conv1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1, stride=1, if_bn=True, activation_fn=torch.relu):
        super(Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride=stride)
        self.if_bn = if_bn
        self.bn = nn.BatchNorm1d(out_channel)
        self.activation_fn = activation_fn

    def forward(self, x):
        out = self.conv(x)
        if self.if_bn:
            out = self.bn(out)

        if self.activation_fn is not None:
            out = self.activation_fn(out)

        return out


def random_subsample(pcd, n_points=2048):
    """
    Args:
        pcd: (B, N, 3)

    returns:
        new_pcd: (B, n_points, 3)
    """
    b, n, _ = pcd.shape
    device = pcd.device
    batch_idx = torch.arange(b, dtype=torch.long, device=device).reshape((-1, 1)).repeat(1, n_points)
    idx = torch.cat([torch.randperm(n, dtype=torch.long, device=device)[:n_points].reshape((1, -1)) for i in range(b)],
                    0)
    return pcd[batch_idx, idx, :]


############################################################
# PointNet++ utils
def pc_normalize(pc, norm='instance'):
    """
    Batch Norm to Instance Norm
    Normalize Point Clouds | Pytorch Version | Range: [-1, 1]
    """
    points = pc[:, :3, :]
    centroid = torch.mean(points, dim=2, keepdim=True)
    points = points - centroid
    if norm == 'instance':
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)), dim=1)[0]
        pc[:, :3, :] = points / m.view(-1, 1, 1)
    else:
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
        pc[:, :3, :] = points / m
    return pc


def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def euclidean_distance(src, dst):
    """
    Calculate Euclidean distance

    """
    return torch.norm(src.unsqueeze(-2) - dst.unsqueeze(-3), p=2, dim=-1)


def index_points(points, idx, cuda=False, is_group=False):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    if cuda:
        if is_group:
            points = grouping_operation(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 3, 1).contiguous()
        else:
            points = gather_operation(points.transpose(1, 2).contiguous(), idx)
            return points.permute(0, 2, 1).contiguous()
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, cuda=False):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]

    FLOPs:
        S * (3 + 3 + 2)
    """
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        return furthest_point_sample(xyz, npoint)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, debug=False, cuda=False):
    if cuda:
        if not xyz.is_contiguous():
            xyz = xyz.contiguous()
        if not new_xyz.is_contiguous():
            new_xyz = new_xyz.contiguous()
        return ball_query(radius, nsample, xyz, new_xyz)
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    if debug:
        num_miss = torch.sum(mask)
        num_over = torch.sum(torch.clamp(torch.sum(sqrdists < radius ** 2, dim=2) - nsample, min=0))
        return num_miss, num_over
    return group_idx


# def query_knn_point(xyz, new_xyz, cuda=False):
#     if cuda:
#         if not xyz.is_contiguous():
#             xyz = xyz.contiguous()
#         if not new_xyz.is_contiguous():
#             new_xyz = new_xyz.contiguous()
#         with torch.no_grad():
#             _, idx = repsurfKnn(xyz, new_xyz)  # bs k np
#         return idx
#     dist = square_distance(new_xyz, xyz)
#     group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :9]
#     return group_idx


def sample(nsample, feature, cuda=False):
    feature = feature.permute(0, 2, 1)
    xyz = feature[:, :, :3]

    fps_idx = farthest_point_sample(xyz, nsample, cuda=cuda)  # [B, npoint, C]
    torch.cuda.empty_cache()
    feature = index_points(feature, fps_idx, cuda=cuda, is_group=False)
    torch.cuda.empty_cache()
    feature = feature.permute(0, 2, 1)

    return feature


def fps_sample_points(nsample, points):
    """
    fps
    :param nsample:
    :param points: (b, n, 3)
    :return: (b, n, 3)
    """
    fps_idx = farthest_point_sample(points, nsample, cuda=True)
    res_points = index_points(points, fps_idx, cuda=True, is_group=False)

    return res_points

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = furthest_point_sample(data, number) 
    fps_data = gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


############################################################
# # Basic modules
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.):
        super(CBL, self).__init__()
        self.drop_rate = drop_rate
        self.cbl = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                                 nn.BatchNorm1d(out_channels),
                                 nn.LeakyReLU(negative_slope=0.2))
        if drop_rate != 0.:
            self.drop_out = nn.Dropout(drop_rate)

    def forward(self, x):
        if self.drop_rate == 0.:
            return self.cbl(x)
        else:
            return self.drop_out(self.cbl(x))


# class EdgeConv(nn.Module):
#     def __init__(self, input_channels, mlp, out_npoints=None, aggr_type='max'):
#         super(EdgeConv, self).__init__()
#         self.out_npoints = out_npoints
#         self.aggr_type = aggr_type

#         last_channels = input_channels * 2
#         self.mlp_conv = []
#         for out_channel in mlp:
#             self.mlp_conv.append(nn.Conv2d(last_channels, out_channel, kernel_size=(1, 1), stride=(1, 1)))
#             self.mlp_conv.append(nn.GroupNorm(4, out_channel))
#             self.mlp_conv.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
#             last_channels = out_channel
#         self.mlp_conv = nn.Sequential(*self.mlp_conv)

#     @staticmethod
#     def get_graph_feature(coor_q, x_q, coor_k, x_k):

#         # coor: bs, 3, np, x: bs, c, np

#         batch_size = x_k.size(0)
#         num_points_k = x_k.size(2)
#         num_points_q = x_q.size(2)

#         with torch.no_grad():
#             _, idx = edgeKnn(coor_k, coor_q)  # bs k np
#             assert idx.shape[1] == k
#             idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
#             idx = idx + idx_base
#             idx = idx.view(-1)
#         num_dims = x_k.size(1)
#         x_k = x_k.transpose(2, 1).contiguous()
#         feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
#         feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
#         x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
#         feature = torch.cat((feature - x_q, x_q), dim=1)
#         return feature

#     @staticmethod
#     def fps_downsample(coor, x, num_group):
#         xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
#         fps_idx = furthest_point_sample(xyz, num_group)

#         combined_x = torch.cat([coor, x], dim=1)

#         new_combined_x = (
#             gather_operation(
#                 combined_x, fps_idx
#             )
#         )

#         new_coor = new_combined_x[:, :3].contiguous()
#         new_x = new_combined_x[:, 3:].contiguous()

#         return new_coor, new_x

#     def forward(self, features, points):
#         if self.out_npoints is None:
#             graph_features = self.get_graph_feature(points, features, points, features)
#             features = self.mlp_conv(graph_features)
#             # (batch_size, input_channels*2, num_points, k) -> (batch_size, output_channel, num_points, k)
#             if self.aggr_type == 'max':
#                 out_features = features.max(dim=-1, keepdim=False)[0]
#             elif self.aggr_type == 'avg':
#                 out_features = torch.mean(features, dim=-1)
#             else:
#                 out_features = torch.sum(features, dim=-1)
#             # (batch_size, output_channel, num_points, k) -> (batch_size, out_channels, num_points)
#             return out_features
#         else:
#             new_points, new_features = self.fps_downsample(points, features, self.out_npoints)
#             graph_features = self.get_graph_feature(new_points, new_features, points, features)
#             # (batch_size, input_channels, num_points) -> (batch_size, input_channels*2, num_points, k)
#             features = self.mlp_conv(graph_features)
#             # (batch_size, input_channels*2, num_points, k) -> (batch_size, output_channel, num_points, k)
#             # aggregation
#             if self.aggr_type == 'max':
#                 out_features = features.max(dim=-1, keepdim=False)[0]
#             elif self.aggr_type == 'avg':
#                 out_features = torch.mean(features, dim=-1)
#             else:
#                 out_features = torch.sum(features, dim=-1)
#             # (batch_size, output_channel, num_points, k) -> (batch_size, out_channels, num_points)
#             return out_features, new_points


# class FeatureFuse(nn.Module):
#     def __init__(self, in_channel):
#         super(FeatureFuse, self).__init__()
#
#         self.mlp = nn.Sequential(nn.Conv1d(in_channel*2, in_channel, 1),
#                                  nn.BatchNorm1d(in_channel),
#                                  nn.LeakyReLU(negative_slope=0.2),
#                                  nn.Conv1d(in_channel, in_channel, 1),
#                                  nn.BatchNorm1d(in_channel),
#                                  nn.LeakyReLU(negative_slope=0.2),
#                                  )
#
#     def forward(self, cur_fea, prev_fea):
#         """
#         Args:
#             cur_fea: Tensor, (B, in_channel, N)
#             prev_fea: Tensor, (B, in_channel, N)
#
#         Returns:
#             h: Tensor, (B, in_channel, N)
#         """
#         cat_features = torch.cat([cur_fea, prev_fea], dim=1)
#         out_features = self.mlp(cat_features)
#         return out_features


class FeatureFuse(nn.Module):
    def __init__(self, in_channel):
        super(FeatureFuse, self).__init__()

        self.conv_z = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_r = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.sigmoid)
        self.conv_h = Conv1d(in_channel * 2, in_channel, if_bn=True, activation_fn=torch.relu)

    def forward(self, cur_fea, prev_fea):
        """
        Args:
            cur_fea: Tensor, (B, in_channel, N)
            prev_fea: Tensor, (B, in_channel, N)

        Returns:
            h: Tensor, (B, in_channel, N)
        """
        cat_features = torch.cat([cur_fea, prev_fea], dim=1)
        z = self.conv_z(cat_features)
        r = self.conv_r(cat_features)
        h_hat = self.conv_h(torch.cat([cur_fea, r * prev_fea], 1))
        h = (1 - z) * cur_fea + z * h_hat
        return h


# class point2feature(nn.Module):
#     def __init__(self, out_npoints):
#         super().__init__()
#         self.input_trans = nn.Conv1d(3, 8, 1)
#         self.edge_trans = EdgeConv(input_channels=8, mlp=[8, 8, 16], out_npoints=out_npoints)


#     def forward(self, in_points):
#         raw_feature = self.input_trans(in_points)
#         out_feature, _ = self.edge_trans(raw_feature, in_points)

#         return out_feature

# # Channels Attention Feature Fuse
# class CAFFBlock(nn.Module):
#     def __init__(self, channel, reduction=2):
#         super().__init__()
#         in_channels = channel * 2
#         # cat feature channel --> channel weight
#         self.CALayer = nn.Sequential(
#             nn.Conv1d(in_channels, in_channels//2, kernel_size=1, bias=False),
#             nn.BatchNorm1d(in_channels//2),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(in_channels//2, in_channels, kernel_size=1, bias=False),
#             nn.BatchNorm1d(in_channels),
#             nn.Sigmoid(),
#         )
#         self.fuse_conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
#                                        nn.BatchNorm1d(in_channels),
#                                        nn.LeakyReLU(negative_slope=0.2))
#         self.conv_ratio = Conv1d(in_channels, in_channels, if_bn=True, activation_fn=torch.sigmoid)
#         self.reduce_dim = CBL(in_channels, channel)
        
    
#     def forward(self, fea1, fea2):
#         cat_features = torch.cat([fea1, fea2], dim=1)
#         channel_weight = self.CALayer(cat_features)
#         fuse_feature = self.fuse_conv(cat_features * channel_weight)
#         ratio = self.conv_ratio(fuse_feature)
#         reserve_feature = (1-ratio) * cat_features + ratio * fuse_feature
#         res = self.reduce_dim(reserve_feature)
        
#         return res


# Channels Attention Feature Fuse
class CAFFBlock(nn.Module):
    def __init__(self, channel, reduction=2):
        super().__init__()
        in_channels = channel * 2
        # cat feature channel --> channel weight
        self.CALayer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # global average pooling: point feature --> channel feature
            nn.Conv1d(in_channels, in_channels//2, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels//2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_channels//2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid(),
        )

        self.reduce_dim = CBL(in_channels, channel)
    
    def forward(self, fea1, fea2):
        cat_features = torch.cat([fea1, fea2], dim=1)
        channel_weight = self.CALayer(cat_features)
        channel_attention = cat_features * channel_weight
        # res = self.reduce_dim(channel_attention)
        res = self.reduce_dim(channel_attention)
        
        return res


class FCFuseBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        in_channels = channel * 2
        self.fuse_conv = CBL(in_channels, channel)

    def forward(self, fea1, fea2):
        cat_features = torch.cat([fea1, fea2], dim=1)
        res = self.fuse_conv(cat_features)

        return res


class Fold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=256):
        super().__init__()
        self.in_channel = in_channel
        if step == 1:
            self.num_sample = 2
            self.folding_seed = torch.tensor([[0, 1], [0, 1]]).cuda()
        else:
            self.num_sample = step * step
            a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
            b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
            self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x):
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, self.num_sample)
        seed = self.folding_seed.view(1, 2, self.num_sample).expand(bs, 2, self.num_sample).to(x.device)
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


# class Fold(nn.Module):
#     def __init__(self, in_channel, step, hidden_dim=256):
#         super().__init__()
#         self.in_channel = in_channel
#         self.step = step
#
#         a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
#         b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
#         self.folding_seed = torch.cat([a, b], dim=0).cuda()
#
#         self.folding1 = nn.Sequential(
#             nn.Conv1d(in_channel + 2, hidden_dim, 1),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=False),
#             nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(inplace=False),
#             nn.Conv1d(hidden_dim // 2, 3, 1),
#         )
#
#         self.folding2 = nn.Sequential(
#             nn.Conv1d(in_channel + 3, hidden_dim, 1),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=False),
#             nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(inplace=False),
#             nn.Conv1d(hidden_dim // 2, 3, 1),
#         )
#
#     def forward(self, x):
#         num_sample = self.step * self.step
#         bs = x.size(0)
#         features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
#         seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)
#
#         x = torch.cat([seed, features], dim=1)
#         fd1 = self.folding1(x)
#         x = torch.cat([fd1, features], dim=1)
#         fd2 = self.folding2(x)
#
#         return fd2


class PointNetSA(nn.Module):
    def __init__(self, out_npoints, in_channel, mlp, nsample=32, radius=0.2):
        super(PointNetSA, self).__init__()
        self.out_npoints = out_npoints
        self.nsample = nsample
        self.radius = radius
        self.mlp = mlp

        last_channel = in_channel + 3
        self.mlp_conv = []
        for out_channel in mlp:
            self.mlp_conv.append(nn.Conv2d(last_channel, out_channel, kernel_size=(1, 1), stride=(1, 1)))
            self.mlp_conv.append(nn.BatchNorm2d(out_channel))
            self.mlp_conv.append(nn.LeakyReLU(negative_slope=0.2))
            last_channel = out_channel

        self.mlp_conv = nn.Sequential(*self.mlp_conv)

    @staticmethod
    def sample_and_group(xyz, features, npoint, nsample, radius, use_xyz=True):
        """
        Args:
            xyz: Tensor, (B, 3, N)
            features: Tensor, (B, f, N)
            npoint: int
            nsample: int
            radius: float
            use_xyz: boolean

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_features: Tensor, (B, 3 | f+3 | f, npoint, nsample)
            idx_local: Tensor, (B, npoint, nsample)
            grouped_xyz: Tensor, (B, 3, npoint, nsample)

        """
        xyz_flipped = xyz.permute(0, 2, 1).contiguous()  # (B, N, 3)
        new_xyz = gather_operation(xyz, furthest_point_sample(xyz_flipped, npoint))  # (B, 3, npoint)

        idx = ball_query(radius, nsample, xyz_flipped, new_xyz.permute(0, 2, 1).contiguous())  # (B, npoint, nsample)
        grouped_xyz = grouping_operation(xyz, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.unsqueeze(3).repeat(1, 1, 1, nsample)

        if features is not None:
            grouped_features = grouping_operation(features, idx)  # (B, f, npoint, nsample)
            if use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], 1)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_xyz, new_features, idx, grouped_xyz

    def forward(self, features, points):
        """
        Args:
            features: Tensor, (B, f, N)
            points: Tensor, (B, 3, N)

        Returns:
            new_xyz: Tensor, (B, 3, npoint)
            new_feature: Tensor, (B, mlp[-1], npoint)
        """
        new_xyz, new_features, idx, grouped_xyz = self.sample_and_group(points, features, self.out_npoints,
                                                                        self.nsample, self.radius)
        new_features = self.mlp_conv(new_features)
        new_features = torch.max(new_features, 3)[0]

        return new_features, new_xyz


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
            xyz1: Tensor, (B, 3, N)
            xyz2: Tensor, (B, 3, M)
            points1: Tensor, (B, p1_channels, N)
            points2: Tensor, (B, p2_channels, M)

        Returns:
            new_points: Tensor, (B, mlp[-1], N)
        """
        dist, idx = three_nn(xyz1.permute(0, 2, 1).contiguous(), xyz2.permute(0, 2, 1).contiguous())
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
#########################################################################


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # 1 for mask, 0 for not mask
            # mask shape N, N
            mask_value = -torch.finfo(attn.dtype).max
            mask = (mask > 0)  # convert to boolen, shape torch.BoolTensor[N, N]
            attn = attn.masked_fill(mask, mask_value) # B h N N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DeformableLocalAttention(nn.Module):
    r''' DeformabelLocalAttention for only self attn
        Query a local region for each token (k x C)
        Conduct the Self-Attn among them and use the region feat after maxpooling to update the token feat
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
    def forward(self, x, pos, idx=None):
        B, N, C = x.shape
        # given N token and pos
        assert len(pos.shape) == 3 and pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for pos, expect it to be B N 3, but got {pos.shape}'
        # first query a neighborhood for one query token
        if idx is None:
            idx = knn_point(self.k, pos, pos) # B N k 
        assert idx.size(-1) == self.k
        # project the qeury feat into shared space
        q = self.proj_q(x)
        v_off = self.proj_v_off(x)
        # Then we extract the region feat for a neighborhood
        local_v = index_points(v_off, idx) # B N k C 
        # And we split it into several group on channels
        off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
        group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # calculate offset
        shift_feat = torch.cat([
            off_local_v,
            group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
        ], dim=-1)  # Bg N k 2c
        offset  = self.linear_offset(shift_feat) # Bg N k 3
        offset = offset.tanh() # Bg N k 3
        # add offset for each point
        # The position in R3 for these points
        local_v_pos = index_points(pos, idx) # B N k 3     
        local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
        local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
        shift_pos = local_v_pos + offset # Bg N 2*k 3
        # interpolate
        shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
        pos = pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g N 3  
        pos = einops.rearrange(pos, 'b g n c -> (b g) n c') # Bg N 3
        v = einops.rearrange(x, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
        # three_nn and three_interpolate
        dist, _idx = three_nn(shift_pos.contiguous(), pos.contiguous())  #  Bg k*N 3, Bg k*N 3
        dist_reciprocal = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
        weight = dist_reciprocal / norm
        interpolated_feats = three_interpolate(v.transpose(-1, -2).contiguous(), _idx, weight).transpose(-1, -2).contiguous() 
        interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

        # some assert to ensure the right feature shape
        assert interpolated_feats.size(1) == local_v.size(1)
        assert interpolated_feats.size(2) == local_v.size(2)
        assert interpolated_feats.size(3) == local_v.size(3)
        # SE module to select 1/2k out of k
        pass

        # calculate local attn
        # local_q : B N k C 
        # interpolated_feats : B N k C 
        # extrate the feat for a local region
        local_q = index_points(q, idx) # B N k C
        q = einops.rearrange(local_q, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        k = self.proj_k(interpolated_feats)
        k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
        v = self.proj_v(interpolated_feats)
        v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

        attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, k, k
        attn = attn.mul(self.scale)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN k c
        out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N k C
        out = out.max(dim=2, keepdim=False)[0]  # B N C
        out = self.proj(out)
        out = self.proj_drop(out)

        assert out.size(0) == B
        assert out.size(1) == N
        assert out.size(2) == C

        return out
    

class DeformableLocalCrossAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a cross attn among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., k=10, n_group=2):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_v_off = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Deformable related
        self.k = k  # To be controlled 
        self.n_group = n_group
        self.group_dims = dim // self.n_group
        assert num_heads % self.n_group == 0
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.group_dims, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            k = v
            NK = k.size(1)
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            # project the qeury feat into shared space
            q = self.proj_q(q)
            v_off = self.proj_v_off(v)
            # Then we extract the region feat for a neighborhood
            local_v = index_points(v_off, idx) # B N k C 
            # And we split it into several group on channels
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset  = self.linear_offset(shift_feat) # Bg N k 3
            offset = offset.tanh() # Bg N k 3
            # add offset for each point
            # The position in R3 for these points
            local_v_pos = index_points(v_pos, idx) # B N k 3     
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
            shift_pos = local_v_pos + offset # Bg N k 3
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g Nk 3  
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c') # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg Nk c
            # three_nn and three_interpolate
            dist, idx = three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

            # some assert to ensure the right feature shape
            assert interpolated_feats.size(1) == local_v.size(1)
            assert interpolated_feats.size(2) == local_v.size(2)
            assert interpolated_feats.size(3) == local_v.size(3)
            # SE module to select 1/2k out of k
            pass

            # calculate local attn
            # local_q : B N k C 
            # interpolated_feats : B N k C 
            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(-2) # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N 1 C
            assert out.size(2) == 1
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
            
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            q = self.proj_q(q)
            v_off = self.proj_v_off(v)

            ######################################### produce local_v by two knn #########################################
            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r = index_points(v_off[:, :-denoise_length], idx) # B N_r k C 
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx) # B N_r k 3     
           
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n = index_points(v_off, idx) # B N_n k C 
            local_v_n_pos = index_points(v_pos, idx) # B N_n k 3     
            ######################################### produce local_v by two knn #########################################
            
            # Concat two part
            local_v = torch.cat([local_v_r, local_v_n], dim=1) # B N k C 

            # And we split it into several group on channels
            off_local_v = einops.rearrange(local_v, 'b n k (g c) -> (b g) n k c', g=self.n_group, c=self.group_dims) # Bg N k c
            group_q = einops.rearrange(q, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg N c
            
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                group_q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # Bg N k 2c
            offset  = self.linear_offset(shift_feat) # Bg N k 3
            offset = offset.tanh() # Bg N k 3
            # add offset for each point
            # The position in R3 for these points
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3
            local_v_pos = local_v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1, -1) # B g N k 3  
            local_v_pos = einops.rearrange(local_v_pos, 'b g n k c -> (b g) n k c') # Bg N k 3
            shift_pos = local_v_pos + offset # Bg N k 3
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'bg n k c -> bg (n k) c') # Bg k*N 3
            v_pos = v_pos.unsqueeze(1).expand(-1, self.n_group, -1, -1) # B g Nk 3  
            v_pos = einops.rearrange(v_pos, 'b g n c -> (b g) n c') # Bg Nk 3
            v = einops.rearrange(v, 'b n (g c) -> (b g) n c', g=self.n_group, c=self.group_dims) # Bg Nk c
            # three_nn and three_interpolate
            dist, idx = three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  Bg k*N 3, Bg k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, '(b g) (n k) c  -> b n k (g c)', b=B, g=self.n_group, n=N, k=self.k) # B N k gc

            # some assert to ensure the right feature shape
            assert interpolated_feats.size(1) == local_v.size(1)
            assert interpolated_feats.size(2) == local_v.size(2)
            assert interpolated_feats.size(3) == local_v.size(3)
            # SE module to select 1/2k out of k
            pass

            # calculate local attn
            # local_q : B N k C 
            # interpolated_feats : B N k C 
            q = einops.rearrange(q, 'b n (h c) -> (b h n) c', h=self.num_heads, c=self.head_dim).unsqueeze(-2) # BHN 1 c
            k = self.proj_k(interpolated_feats)
            k = einops.rearrange(k, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c
            v = self.proj_v(interpolated_feats)
            v = einops.rearrange(v, 'b n k (h c) -> (b h n) k c', h=self.num_heads, c=self.head_dim) # BHN k c

            attn = torch.einsum('b m c, b n c -> b m n', q, k) # BHN, 1, k
            attn = attn.mul(self.scale)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = torch.einsum('b m n, b n c -> b m c', attn, v) # BHN 1 c
            out = einops.rearrange(out, '(b h n) k c -> b n k (h c)', b=B, n=N, h=self.num_heads) # B N 1 C
            assert out.size(2) == 1
            out = out.squeeze(2)
            out = self.proj(out)
            out = self.proj_drop(out)

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
            
        return out


class DynamicGraphAttention(nn.Module):
    r''' DynamicGraphAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform Conv2d with maxpooling to build the graph feature for each token
        These can convert local self-attn as a local cross-attn
    '''
    def __init__(self, dim, k=10):
        super().__init__()
        self.dim = dim
        # Deformable related
        self.k = k  # To be controlled 
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v = index_points(v, idx) # B N k C 
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((local_v - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r = index_points(v[:, :-denoise_length], idx) # B N_r k C 
            
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n = index_points(v, idx) # B N_n k C 

            # Concat two part
            local_v = torch.cat([local_v_r, local_v_n], dim=1)
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((local_v - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out


class improvedDeformableLocalGraphAttention(nn.Module):
    r''' DeformabelLocalAttention for self attn or cross attn
        Query a local region for each token (k x C) and then perform a graph conv among query token(1 x C) and local region (k x C)
        These can convert local self-attn as a local cross-attn
        $ improved:
            Deformable within a local ball
    '''
    def __init__(self, dim, k=10):
        super().__init__()
        self.dim = dim

        self.proj_v_off = nn.Linear(dim, dim)

        # Deformable related
        self.k = k  # To be controlled 
        self.linear_offset = nn.Sequential(
            nn.Linear(2 * self.dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 3, bias=False)
        )
        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, q, q_pos, v=None, v_pos=None, idx=None, denoise_length=None):
        r'''
            If perform a self-attn, just use 
                q = x, v = x, q_pos = pos, v_pos = pos
        '''
        if denoise_length is None:
            if v is None:
                v = q
            if v_pos is None:
                v_pos = q_pos

            B, N, C = q.shape
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            # first query a neighborhood for one query token
            if idx is None:
                idx = knn_point(self.k, v_pos, q_pos) # B N k 
            assert idx.size(-1) == self.k
            # project the local feat into shared space
            v_off = self.proj_v_off(v)
            # Then we extract the region feat for a neighborhood
            off_local_v = index_points(v_off, idx) # B N k C 
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset  = self.linear_offset(shift_feat) # B N k 3
            offset = offset.tanh() # B N k 3

            # add offset for each point
            # The position in R3 for these points
            local_v_pos = index_points(v_pos, idx) # B N k 3     

            # calculate scale
            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0] # B N 3
            scale = scale.unsqueeze(-2) * 0.5 # B N 1 3
            shift_pos = local_v_pos + offset * scale # B N k 3
            
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c') # B k*N 3
            # three_nn and three_interpolate
            dist, idx = three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k) # B N k c

            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        
        else:
            assert idx is None, f'we need online index calculation when denoise_length is set, denoise_length {denoise_length}'
            # when v_pos and v are given, that to say, it's a cross attn.
            # we only consider self-attn
            assert v is None, f'mask for denoise_length is only consider in self-attention, but v is given'
            assert v_pos is None, f'mask for denoise_length is only consider in self-attention, but v_pos is given'

            v = q
            v_pos = q_pos
            # given N token and pos
            assert len(v_pos.shape) == 3 and v_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for v_pos, expect it to be B N 3, but got {v_pos.shape}'
            assert len(q_pos.shape) == 3 and q_pos.size(-1) == 3, f'[ERROR] Got an unexpected shape for q_pos, expect it to be B N 3, but got {q_pos.shape}'
            assert q.size(-1) == v.size(-1) == self.dim
            B, N, C = q.shape

            v_off = self.proj_v_off(v)

            # normal reconstruction task:
            # first query a neighborhood for one query token for normal part
            idx = knn_point(self.k, v_pos[:, :-denoise_length], q_pos[:, :-denoise_length]) # B N_r k 
            assert idx.size(-1) == self.k
            # gather the neighbor point feat
            local_v_r_off = index_points(v_off[:, :-denoise_length], idx) # B N_r k C 
            local_v_r_pos = index_points(v_pos[:, :-denoise_length], idx) # B N_r k 3     
            # Then query a nerighborhood for denoise token within all token
            idx = knn_point(self.k, v_pos, q_pos[:, -denoise_length:]) # B N_n k 
            assert idx.size(-1) == self.k
            assert idx.size(1) == denoise_length
            # gather the neighbor point feat
            local_v_n_off = index_points(v_off, idx) # B N_n k C 
            local_v_n_pos = index_points(v_pos, idx) # B N_n k 3     
            # Concat two part
            off_local_v = torch.cat([local_v_r_off, local_v_n_off], dim=1) # B N k C 
            # calculate offset
            shift_feat = torch.cat([
                off_local_v,
                q.unsqueeze(-2).expand(-1, -1, self.k, -1)
            ], dim=-1)  # B N k 2c
            offset  = self.linear_offset(shift_feat) # B N k 3
            offset = offset.tanh() # B N k 3

            # add offset for each point
            # The position in R3 for these points
            local_v_pos = torch.cat([local_v_r_pos, local_v_n_pos], dim=1)  # B N k 3

            # calculate scale
            scale = local_v_pos.max(-2)[0] - local_v_pos.min(-2)[0] # B N 3
            scale = scale.unsqueeze(-2) * 0.5 # B N 1 3
            shift_pos = local_v_pos + offset * scale # B N k 3
            
            # interpolate
            shift_pos = einops.rearrange(shift_pos, 'b n k c -> b (n k) c') # B k*N 3
            # three_nn and three_interpolate
            dist, idx = three_nn(shift_pos.contiguous(), v_pos.contiguous())  #  B k*N 3, B k*N 3
            dist_reciprocal = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_reciprocal, dim=2, keepdim=True)
            weight = dist_reciprocal / norm
            interpolated_feats = three_interpolate(v.transpose(-1, -2).contiguous(), idx, weight).transpose(-1, -2).contiguous() 
            interpolated_feats = einops.rearrange(interpolated_feats, 'b (n k) c  -> b n k c', n=N, k=self.k) # B N k c
            
            q = q.unsqueeze(-2).expand(-1, -1, self.k, -1) # B N k C
            feature = torch.cat((interpolated_feats - q, q), dim=-1) # B N k C
            out = self.knn_map(feature).max(-2)[0] # B N C

            assert out.size(0) == B
            assert out.size(1) == N
            assert out.size(2) == C
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class CrossAttention(nn.Module):
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
        return x


class MLP_CONV(nn.Module):
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128,bn=False):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        if bn:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = None

        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        if self.bn:
            out = self.conv_2(torch.relu(self.bn(self.conv_1(x)))) + shortcut
        else:
            out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


