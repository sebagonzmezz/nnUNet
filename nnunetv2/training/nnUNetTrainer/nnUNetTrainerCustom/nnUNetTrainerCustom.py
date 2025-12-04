from typing import Tuple, Union, List

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn
import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.SwinTransformer import SwinTransformerV2, SwinTransformerV3, SwinTransformerV4

class G2_in(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   pretrained_encoder = None) -> nn.Module:
        model = PlainConvUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[1, 48, 96, 192, 384, 384],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=3,
            strides=[1, 4, 2, 2, 2, 1],
            n_conv_per_stage=2,
            num_classes=2,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05,"affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=enable_deep_supervision,
        )
        encoder = SwinTransformerV2(
            in_chans=1,
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(4, 4, 4),
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            spatial_dims=3,
            return_all_tokens=False,
            masked_im_modeling=True,
            swin_weights_path=pretrained_encoder,
            freeze_swin=True,
        )
        print(f"Using pretrained encoder weights from: {pretrained_encoder}")
        model.encoder = encoder
        total_params = sum(p.numel() for p in model.encoder.parameters())
        trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"Total parameters in encoder: {total_params:,}")
        print(f"Trainable parameters in encoder: {trainable_params:,}")
        return model
    
class G2_enc(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   pretrained_encoder = None) -> nn.Module:
        model = PlainConvUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[24, 48, 96, 192, 384, 384],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=3,
            strides=[1, 4, 2, 2, 2, 1],
            n_conv_per_stage=2,
            num_classes=2,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05,"affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=enable_deep_supervision,
        )
        encoder = SwinTransformerV3(
            conv_output_channels=24,
            in_chans=1,
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(4, 4, 4),
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            spatial_dims=3,
            return_all_tokens=False,
            masked_im_modeling=True,
            swin_weights_path=pretrained_encoder,
            freeze_swin=True,
        )
        print(f"Using pretrained encoder weights from: {pretrained_encoder}")
        model.encoder = encoder
        total_params = sum(p.numel() for p in model.encoder.parameters())
        trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"Total parameters in encoder: {total_params:,}")
        print(f"Trainable parameters in encoder: {trainable_params:,}")
        return model

class G2_up(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   pretrained_encoder = None) -> nn.Module:
        model = PlainConvUNet(
            input_channels=1,
            n_stages=5,
            features_per_stage=[48, 96, 192, 384, 384],
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=3,
            strides=[1, 4, 2, 2, 2],
            n_conv_per_stage=2,
            num_classes=2,
            n_conv_per_stage_decoder=2,
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-05,"affine": True},
            nonlin=torch.nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=enable_deep_supervision,
        )
        encoder = SwinTransformerV4(
            in_chans=1,
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(4, 4, 4),
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            spatial_dims=3,
            return_all_tokens=False,
            masked_im_modeling=True,
            swin_weights_path=pretrained_encoder,
            freeze_swin=True,
        )
        print(f"Using pretrained encoder weights from: {pretrained_encoder}")
        model.encoder = encoder
        total_params = sum(p.numel() for p in model.encoder.parameters())
        trainable_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
        print(f"Total parameters in encoder: {total_params:,}")
        print(f"Trainable parameters in encoder: {trainable_params:,}")
        return model
    
class random_encoder(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        model = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )
        for param in model.encoder.parameters():
            param.requires_grad = False
        return model