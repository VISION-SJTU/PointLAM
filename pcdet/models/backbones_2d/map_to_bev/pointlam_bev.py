import torch
import torch.nn as nn


class RandomTopKBEV(nn.Module):
    """Project points to BEV and randomly retain up to top_k features per cell."""

    def __init__(self, voxel_size, point_cloud_range, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.x_min, self.x_max = point_cloud_range[0], point_cloud_range[3]
        self.y_min, self.y_max = point_cloud_range[1], point_cloud_range[4]

        self.out_height = round((self.y_max - self.y_min) / self.voxel_y)
        self.out_width = round((self.x_max - self.x_min) / self.voxel_x)
        self.x_scale = self.out_width / max((self.x_max - self.x_min), 1e-5)
        self.y_scale = self.out_height / max((self.y_max - self.y_min), 1e-5)

    def forward(self, coords, features, indices):
        batch_size = torch.max(indices) + 1
        _, channels = features.shape
        device = features.device

        # Discretize point coordinates onto the BEV grid.
        x_bev = (
            ((coords[:, 0] - self.x_min) * self.x_scale)
            .long()
            .clamp(min=0, max=self.out_width - 1)
        )
        y_bev = (
            ((coords[:, 1] - self.y_min) * self.y_scale)
            .long()
            .clamp(min=0, max=self.out_height - 1)
        )
        bev_index_1d = (
            indices * (self.out_height * self.out_width)
            + y_bev * self.out_width
            + x_bev
        )

        # Keep random slot order when grouping points by BEV cell.
        num_points = features.shape[0]
        rand_perm = torch.randperm(num_points, device=features.device)
        sorted_features = features[rand_perm]
        sorted_bev_index = bev_index_1d[rand_perm]

        final_bev_index, final_perm = torch.sort(sorted_bev_index, stable=True)
        final_features = sorted_features[final_perm]

        num_points = final_bev_index.shape[0]
        point_range = torch.arange(num_points, device=device)
        _, inverse_indices, counts = torch.unique(
            final_bev_index, return_inverse=True, return_counts=True
        )
        group_end_indices = torch.cumsum(counts, dim=0)
        group_start_indices = torch.cat(
            [torch.tensor([0], device=device), group_end_indices[:-1]]
        )
        point_group_starts = group_start_indices[inverse_indices]
        local_rank = point_range - point_group_starts
        topk_mask = local_rank < self.top_k

        topk_features = final_features[topk_mask]
        topk_bev_indices = final_bev_index[topk_mask]
        topk_local_rank = local_rank[topk_mask]

        output_flat = torch.zeros(
            batch_size * self.out_height * self.out_width,
            self.top_k,
            channels,
            device=device,
        )
        output_flat.index_put_((topk_bev_indices, topk_local_rank), topk_features)
        output_flat = output_flat.view(
            batch_size * self.out_height * self.out_width,
            self.top_k * channels,
        )
        output_bev_permuted = output_flat.view(
            batch_size,
            self.out_height,
            self.out_width,
            self.top_k * channels,
        )

        return output_bev_permuted.permute(0, 3, 1, 2).contiguous()


class PointLAMBEV(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        input_feature_dim = self.model_cfg.NUM_BEV_FEATURES

        top_k = self.model_cfg.get("TOP_K", 2)
        self.num_bev_features = input_feature_dim * top_k
        self.bev_projection = RandomTopKBEV(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            top_k=top_k,
        )

    def forward(self, batch_dict):
        features = batch_dict["encoded_spconv_tensor"]
        xyz = batch_dict["points_xyz"]
        indices = batch_dict["points_indice"]

        batch_dict["spatial_features"] = self.bev_projection(
            xyz.contiguous(), features, indices
        )
        batch_dict["spatial_features_stride"] = batch_dict[
            "encoded_spconv_tensor_stride"
        ]

        return batch_dict
