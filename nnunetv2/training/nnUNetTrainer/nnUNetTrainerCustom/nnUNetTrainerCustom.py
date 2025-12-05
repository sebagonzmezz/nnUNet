from typing import Tuple, Union, List

from torch import nn

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
        model = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
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
        model = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
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

class G2_up_all(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True,
                                   pretrained_encoder = None) -> nn.Module:
        model = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
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