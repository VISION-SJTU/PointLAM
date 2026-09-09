import torch
import torch.nn as nn
from mamba_ssm import Mamba

from ...utils.spconv_utils import spconv


class LocalHadamardAggregator(nn.Module):
    """Sparse CPE feature gating."""

    def __init__(
        self,
        in_planes,
        out_planes,
        voxel_size=None,
        point_cloud_range=None,
        grid_size=None,
    ):
        super().__init__()

        self.mid_planes = out_planes

        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.grid_size = torch.tensor(grid_size).cuda()

        self.input_norm = nn.LayerNorm(out_planes)
        self.gate_projection = nn.Linear(in_planes, self.mid_planes)
        self.local_context = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_planes,
                in_planes,
                kernel_size=3,
                bias=False,
                indice_key="subm1",
            ),
            nn.BatchNorm1d(in_planes),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(
                in_planes,
                in_planes,
                kernel_size=3,
                bias=False,
                indice_key="subm1",
            ),
            nn.Linear(in_planes, in_planes),
            nn.BatchNorm1d(in_planes),
        )
        self.output_projection = nn.Linear(self.mid_planes, out_planes)

    def forward(self, x, xyz, indices=None):
        x_flat = x
        batch_size = torch.max(indices) + 1

        # Build a sparse tensor on the current voxel grid for local CPE.
        points_coords = torch.floor(
            (xyz - self.point_cloud_range[[0, 1, 2]]) / self.voxel_size[[0, 1, 2]]
        ).int()
        point_coords = torch.stack(
            (
                indices,
                points_coords[:, 0],
                points_coords[:, 1],
                points_coords[:, 2],
            ),
            dim=1,
        )
        point_coords = point_coords[:, [0, 3, 2, 1]]

        sparse_conv_feat = spconv.SparseConvTensor(
            features=x_flat,
            indices=point_coords,
            spatial_shape=self.grid_size.tolist()[::-1],
            batch_size=batch_size,
        )

        x_k = self.local_context(sparse_conv_feat).features
        x_q = self.gate_projection(self.input_norm(x_flat))
        x_w = self.output_projection(x_k * x_q)

        return x + x_w


class BidirectionalMambaAggregation(nn.Module):
    """Bidirectional Mamba aggregation within each frame's point sequence."""

    def __init__(self, d_model):
        super().__init__()

        d_state = 16
        d_conv = 4
        expand = 2
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, indices):
        if x.shape[0] == 0:
            return x

        # Scan ordering keeps each frame contiguous. Separate calls reset both
        # convolution history and SSM state at every frame boundary.
        _, counts = torch.unique_consecutive(indices, return_counts=True)
        sequences = torch.split(x, counts.tolist(), dim=0)
        out = torch.cat(
            [self.mamba(sequence.unsqueeze(0)).squeeze(0) for sequence in sequences],
            dim=0,
        )
        out = self.dropout(self.output_norm(out))
        return x + out


class LocalAttentiveMambaBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        grid_size=None,
        voxel_size=None,
        point_cloud_range=None,
    ):
        super().__init__()

        self.bdm_x = BidirectionalMambaAggregation(in_channels)
        self.bdm_y = BidirectionalMambaAggregation(in_channels)
        self.lha_x = LocalHadamardAggregator(
            in_channels,
            in_channels,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
        )
        self.lha_y = LocalHadamardAggregator(
            in_channels,
            in_channels,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
        )

    def forward(self, xyz, x, y_index=None, indices=None):
        x = self.bdm_x(x, indices)
        x = self.lha_x(x, xyz, indices=indices)

        # Reorder once for the y-direction Mamba pass, then restore x-order.
        xyz, x = xyz[y_index[0]], x[y_index[0]]
        x = self.bdm_y(x, indices[y_index[0]])
        xyz, x = xyz[y_index[1]], x[y_index[1]]
        x = self.lha_y(x, xyz, indices=indices)

        return xyz, x


class PointLAMBackbone(nn.Module):
    def __init__(
        self,
        model_cfg,
        input_channels,
        grid_size,
        voxel_size,
        point_cloud_range,
        **kwargs,
    ):
        super().__init__()

        self.initial_grid_size = torch.tensor(grid_size).cuda()
        self.initial_voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        # DSS retains at most two points per progressively coarser cell.
        self.dss_top_k = 2

        in_channels = 128

        # LHA always uses the initial fine grid; only DSS and scan ordering coarsen.
        self.lam_block1 = LocalAttentiveMambaBlock(
            in_channels,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.lam_block2 = LocalAttentiveMambaBlock(
            in_channels,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.lam_block3 = LocalAttentiveMambaBlock(
            in_channels,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.lam_block4 = LocalAttentiveMambaBlock(
            in_channels,
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.dss = DoublySortedSampler(top_k=self.dss_top_k)

    def _compute_y_index(self, xyz, indices, grid_size, voxel_size):
        """Build scan-line order for the y-direction pass."""
        points_coords = torch.floor((xyz - self.point_cloud_range[:3]) / voxel_size).int()
        merge_coords = (
            indices.int() * grid_size[0] * grid_size[1] * grid_size[2]
            + points_coords[:, 1] * grid_size[0] * grid_size[2]
            + points_coords[:, 0] * grid_size[2]
            + points_coords[:, 2]
        )

        x2y = torch.argsort(merge_coords, dim=0, stable=True)
        y2x = torch.argsort(x2y)
        return [x2y, y2x]

    def forward(self, batch_dict):
        xyz = batch_dict["points_xyz"]
        features = batch_dict["points_features"]
        indices = batch_dict["points_indice"]

        current_grid_size = self.initial_grid_size.clone()
        current_voxel_size = self.initial_voxel_size.clone()
        z_range = self.point_cloud_range[5] - self.point_cloud_range[2]
        layers = (self.lam_block1, self.lam_block2, self.lam_block3, self.lam_block4)

        for stage_idx, layer in enumerate(layers):
            y_index = self._compute_y_index(
                xyz, indices, current_grid_size, current_voxel_size
            )
            xyz, features = layer(xyz, features, y_index=y_index, indices=indices)

            if stage_idx == len(layers) - 1:
                continue

            current_grid_size[2] = (current_grid_size[2] / 2).to(torch.int64)
            current_voxel_size[2] = z_range / current_grid_size[2]
            xyz, features, indices = self.dss(
                xyz,
                features,
                indices,
                current_voxel_size,
                self.point_cloud_range,
                current_grid_size,
            )

        features = self.feature_projection(features)
        batch_dict.update(
            {
                "encoded_spconv_tensor": features,
                "points_xyz": xyz,
                "points_indice": indices,
                "encoded_spconv_tensor_stride": 1,
            }
        )

        return batch_dict


class DoublySortedSampler(nn.Module):
    """Select top-k points per dynamic 3D grid cell."""

    def __init__(self, top_k=2):
        super().__init__()
        self.top_k = top_k

    def forward(self, xyz, features, indices, voxel_size, point_cloud_range, grid_size):
        device = features.device
        coords_mc = torch.floor((xyz - point_cloud_range[:3]) / voxel_size).int()
        grid_index_1d = (
            indices * (grid_size[0] * grid_size[1] * grid_size[2])
            + coords_mc[:, 0] * grid_size[1] * grid_size[2]
            + coords_mc[:, 1] * grid_size[2]
            + coords_mc[:, 2]
        )

        # Randomize first, then preserve that order within each coarse cell.
        num_points = features.shape[0]
        rand_perm = torch.randperm(num_points, device=features.device)
        sorted_xyz = xyz[rand_perm]
        sorted_features = features[rand_perm]
        sorted_indices = indices[rand_perm]
        sorted_grid_index = grid_index_1d[rand_perm]

        final_grid_index, final_perm = torch.sort(sorted_grid_index, stable=True)
        final_xyz = sorted_xyz[final_perm]
        final_features = sorted_features[final_perm]
        final_indices = sorted_indices[final_perm]

        num_points = final_grid_index.shape[0]
        point_range = torch.arange(num_points, device=device)
        _, inverse_indices, counts = torch.unique(
            final_grid_index, return_inverse=True, return_counts=True
        )
        group_end_indices = torch.cumsum(counts, dim=0)
        group_start_indices = torch.cat(
            [torch.tensor([0], device=device), group_end_indices[:-1]]
        )
        point_group_starts = group_start_indices[inverse_indices]
        local_rank = point_range - point_group_starts
        topk_mask = local_rank < self.top_k

        return final_xyz[topk_mask], final_features[topk_mask], final_indices[topk_mask]
