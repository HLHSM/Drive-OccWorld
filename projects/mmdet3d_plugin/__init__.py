# FarmSim training is image-only and occupancy-only.  Import just the modules
# that register this path; importing the historical nuScenes/DD3D package
# would otherwise require Detectron2 and 3D detection CUDA operators.
from .datasets.farmsim_world_dataset import FarmSimWorldDataset
from .datasets.orad3d_world_dataset import ORAD3DWorldDataset
from .core.hooks.set_epoch_info_hook import SetEpochInfoHook
from .bevformer.modules.transformer import PerceptionTransformer
from .bevformer.modules.spatial_cross_attention import (SpatialCrossAttention,
                                                         MSDeformableAttention3D)
from .bevformer.modules.temporal_self_attention import TemporalSelfAttention
from .bevformer.modules.agri_bev_attention import (
    GeometryVisibleAnchorDeformableAttention, DirectionalDecaySelectiveRetention)
from .bevformer.modules.encoder import BEVFormerEncoder
from .bevformer.modules.decoder import (DetectionTransformerDecoder,
                                        CustomMSDeformableAttention)
from .bevformer.modules.encoder_v2 import (BEVFormerLayerV2,
                                           CustomBEVFormerEncoder)
from .bevformer.modules.conditionalnorm import ConditionalNorm
from .bevformer.modules.group_attention import GroupMultiheadAttention
from .bevformer.modules.world_decoder import (WorldDecoder,
                                              PredictionTransformerLayer,
                                              PredictionMSDeformableAttention)
from .bevformer.modules.world_transformer import PredictionTransformer
from .bevformer.dense_heads.bevformer_head import BEVFormerHead
from .bevformer.dense_heads.world_bevformer_head import WorldBEVFormerHead
from .bevformer.dense_heads.world_head_v1 import WorldHeadV1
# These are only loaded when the trajectory switch is enabled, but keeping
# registration available makes the shell-level trajectory option functional.
from .bevformer.dense_heads.plan_head import PlanHead_v2
from .bevformer.losses.planning_loss import PlanningLoss, CollisionLoss
from .bevformer.detectors.bevformer import BEVFormer
from .bevformer.detectors.drive_occworldV2 import Drive_OccWorld_V2
