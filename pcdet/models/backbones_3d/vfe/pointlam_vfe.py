import torch
import torch.nn as nn

try:
    import torch_scatter
except Exception as e:
    pass

from .vfe_template import VFETemplate


class DeviationLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False,
                 deviation_mode='sub'):
        super().__init__()

        self.is_output_layer = last_layer
        self.use_norm = use_norm
        if deviation_mode != 'sub':
            raise ValueError('PointLAM VFE requires regional subtraction')
        if not self.is_output_layer:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)

        if self.is_output_layer:
            return x
        else:
            x_mean = torch_scatter.scatter_mean(x, unq_inv, dim=0)
            x_context = x - x_mean[unq_inv, :]
            x_concatenated = torch.cat([x, x_context], dim=1)
            return x_concatenated


class PointLAMVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        self.deviation_mode = str(self.model_cfg.get('DEVIATION_MODE', 'sub')).lower()
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        devnet_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            devnet_layers.append(
                DeviationLayer(in_filters, out_filters, self.use_norm,
                           last_layer=(i >= len(num_filters) - 2),
                           deviation_mode=self.deviation_mode)
            )
        self.devnet_layers = nn.ModuleList(devnet_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        self.lps_top_k = float(self.model_cfg.get('TOPK_NUM', 1))
        assert self.lps_top_k == 1, 'PointLAM uses one LPS point per cell'

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    @staticmethod
    def _compute_integer_topk_mask(k, local_voxel_indices):
        k = int(k)
        total_points = local_voxel_indices.shape[0]
        if k <= 0:
            return torch.zeros(total_points, dtype=torch.bool, device=local_voxel_indices.device)

        mask = torch.ones(total_points, dtype=torch.bool, device=local_voxel_indices.device)
        if k < total_points:
            mask[k:] = local_voxel_indices[k:] != local_voxel_indices[:-k]
        return mask

    def _doubly_sorted_sampling(self, points, features, points_xyz, unq_inv, lps_top_k):
        """Select points by feature L2 norm within each batch-specific cell.

        Return features, xyz and batch indices in cell order.
        """

        dim = 0
        indices = points[:, 0].int()

        features_for_sort = torch.linalg.norm(features, ord=2, dim=1)
        index = unq_inv

        _, features_perm = torch.sort(features_for_sort, descending=True, dim=dim)
        features = features[features_perm]
        points_xyz = points_xyz[features_perm]
        indices = indices[features_perm]
        index = index[features_perm]

        # Preserve the descending feature-norm order inside each cell.
        index, index_perm = torch.sort(index, dim=dim, stable=True)
        features = features[index_perm]
        points_xyz = points_xyz[index_perm]
        indices = indices[index_perm]


        topk_value = float(lps_top_k)
        if not topk_value.is_integer():
            raise ValueError('PointLAM LPS requires an integer point count')
        topk_mask = self._compute_integer_topk_mask(int(topk_value), index)

        batch_xyz = points_xyz[topk_mask]
        batch_features = features[topk_mask]
        batch_indice = indices[topk_mask]


        return batch_features, batch_xyz, batch_indice

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points']

        points_coords = torch.floor((points[:, [1,2,3]] - self.point_cloud_range[[0,1,2]]) / self.voxel_size[[0,1,2]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1,2]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        # Include the batch index so cells never mix different frames.
        merge_coords = points[:, 0].int() * self.scale_xyz + \
                       points_coords[:, 0] * self.scale_yz + \
                       points_coords[:, 1] * self.scale_z + \
                       points_coords[:, 2]

        _, unq_inv = torch.unique(merge_coords, return_inverse=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)

        f_center[:, 2] = points_xyz[:, 2] - (points_coords[:, 2].to(points_xyz.dtype) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for devnet_layer in self.devnet_layers:
            features = devnet_layer(features, unq_inv)

        features, points_xyz, indices = self._doubly_sorted_sampling(points, features, points_xyz, unq_inv,
                                                lps_top_k=self.lps_top_k)

        batch_dict['points_features'] = features
        batch_dict['points_xyz'] = points_xyz
        batch_dict['points_indice'] = indices


        return batch_dict
