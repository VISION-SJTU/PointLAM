# Architecture and implementation

| Component | Implementation |
|---|---|
| Laplacian Point Sampler | [PointLAMVFE](../pcdet/models/backbones_3d/vfe/pointlam_vfe.py): mean-subtracted features and one selected point per initial cell |
| Local Attentive Mamba | [PointLAMBackbone](../pcdet/models/backbones_3d/pointlam_backbone.py): four 128-channel blocks with bidirectional Mamba and local Hadamard aggregation |
| Downsampling | Randomly retain up to two points per cell between blocks, with progressively coarser vertical cells |
| BEV projection | [PointLAMBEV](../pcdet/models/backbones_2d/map_to_bev/pointlam_bev.py): concatenate up to two randomly selected features per BEV cell |

Local Hadamard aggregation uses two consecutive 3×3×3 sparse convolutions on the initial fine grid in every block. The initial cell size is `[0.32, 0.32, 0.1875]` meters; only inter-block sampling and scan ordering use the progressively coarser vertical grid.

The code follows the paper's module names: `devnet_layers` implements DevNet inside the LPS encoder; `lam_block1` through `lam_block4` contain the axis-specific `bdm_x`/`bdm_y` and `lha_x`/`lha_y` modules. Initial LPS selection uses `_doubly_sorted_sampling`; inter-block `dss` retains the configured random point order. `bev_projection` maps the final point features to BEV. The `vfe`, `backbone_3d`, and `map_to_bev_module` interfaces follow OpenPCDet's model assembly conventions.

Each bidirectional Mamba pass processes frames separately, resetting convolution history and SSM state at every frame boundary.
