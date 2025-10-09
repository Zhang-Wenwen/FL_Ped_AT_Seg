import torch
import torch.nn as nn
from typing import Sequence, Union, Optional, Tuple, List
from torch.nn.functional import interpolate
from monai.networks.nets.dynunet import DynUNetSkipLayer, DynUNet
from loguru import logger
from torch.nn.functional import softmax

class EncDecDynUNet(nn.Module):
    def __init__(
        self,
        dynunet:DynUNet
    ) -> None:
        """
            Re-implementation of monai.networks.nets.DynUNet.
            The original implementation constructs the model in a recursive manner, not exposing
            necessary encoder and decoder APIs.
            For testing, the class is initialize with a DynUNet block, while aiming to achieve the same
            behavior as the initialization block without calling its `forward()` function.
        """
        super().__init__()

        def _parse_skip_layers(skip_layers:DynUNetSkipLayer):
            super_head, downsample, upsample = nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
            cur_skip_layer = skip_layers
            while hasattr(cur_skip_layer, "next_layer"):
                super_head.append(cur_skip_layer.super_head)
                downsample.append(cur_skip_layer.downsample)
                upsample.append(cur_skip_layer.upsample)
                cur_skip_layer = cur_skip_layer.next_layer
            return super_head, downsample, upsample, cur_skip_layer
        
        self.filters = dynunet.filters

        self.super_heads, self.downsamples, self.upsamples, self.bottleneck = _parse_skip_layers(dynunet.skip_layers)
        self.heads, self.output_block, self.deep_supervision = dynunet.heads, dynunet.output_block, dynunet.deep_supervision

    def down_op(self, x):
        downouts:List[torch.Tensor] = []
        for downsample in self.downsamples:
            x_down = downsample(x)
            downouts.append(x_down)
            x = x_down
        return downouts
    
    def up_op(self, downouts, bottleneckout):
        upouts:List[torch.Tensor] = []
        nextout = bottleneckout
        for up_idx in range(len(self.upsamples)):
            x_up = self.upsamples[-(up_idx + 1)](nextout, downouts[-(up_idx + 1)])
            upouts.append(x_up)
            nextout = x_up
        return upouts[::-1]

    def forward(self, x):
        # downsample
        # logger.info(f"x{x.shape}")

        downouts = self.down_op(x)

        # bottleneck
        bottleneckout = self.bottleneck(downouts[-1])
        # encoder: self.down_op and self.bottleneck


        # upsample
        upouts = self.up_op(downouts, bottleneckout)
        # deepsupervision
        for idx, super_head, upout in zip(range(len(upouts)), self.super_heads, upouts):
            if super_head is not None and self.heads is not None and idx > 0: self.heads[idx - 1] = super_head(upout)

        # the same implementation as in `DynUNet.forward()` below
        out:torch.Tensor = upouts[0]
        out = self.output_block(out)

        if self.training and self.deep_supervision:
            out_all = [out]
            for feature_map in self.heads: out_all.append(interpolate(feature_map, out.shape[2:]))
            return torch.stack(out_all, dim=1)

        return out
    

def create_dynunet(
    in_channels:int,
    out_channels:int,
    spatial_dims:int=3,
    kernel_size:Sequence[Union[Sequence[int], int]]=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides:Sequence[Union[Sequence[int], int]]=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size:Sequence[Union[Sequence[int], int]]=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters:Optional[Sequence[int]]=None,
    dropout:Optional[Union[Tuple, str, float]] = None,
    norm_name:Union[Tuple, str]=("INSTANCE", {"affine": True}),
    act_name:Union[Tuple, str]=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    deep_supervision:bool=False,
    deep_supr_num:int=3,
    res_block:bool=False,
    trans_bias:bool=False,
):
    """
    Wrapper for creating dynamic UNet (DynUNet) re-implemented in encoder-decoder manner.
    See `monai.networks.nets.dynunet.DynUNet` for detailed documentation.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: convolution kernel size.
        strides: convolution strides for each blocks.
        upsample_kernel_size: convolution kernel size for transposed convolution layers. The values should
            equal to strides[1:].
        filters: number of output channels for each blocks. Different from nnU-Net, in this implementation we add
            this argument to make the network more flexible. As shown in the third reference, one way to determine
            this argument is like:
            ``[64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(strides)]``.
            The above way is used in the network that wins task 1 in the BraTS21 Challenge.
            If not specified, the way which nnUNet used will be employed. Defaults to ``None``.
        dropout: dropout ratio. Defaults to no dropout.
        norm_name: feature normalization type and arguments. Defaults to ``INSTANCE``.
            `INSTANCE_NVFUSER` is a faster version of the instance norm layer, it can be used when:
            1) `spatial_dims=3`, 2) CUDA device is available, 3) `apex` is installed and 4) non-Windows OS is used.
        act_name: activation layer type and arguments. Defaults to ``leakyrelu``.
        deep_supervision: whether to add deep supervision head before output. Defaults to ``False``.
            If ``True``, in training mode, the forward function will output not only the final feature map
            (from `output_block`), but also the feature maps that come from the intermediate up sample layers.
            In order to unify the return type (the restriction of TorchScript), all intermediate
            feature maps are interpolated into the same size as the final feature map and stacked together
            (with a new dimension in the first axis)into one single tensor.
            For instance, if there are two intermediate feature maps with shapes: (1, 2, 16, 12) and
            (1, 2, 8, 6), and the final feature map has the shape (1, 2, 32, 24), then all intermediate feature maps
            will be interpolated into (1, 2, 32, 24), and the stacked tensor will has the shape (1, 3, 2, 32, 24).
            When calculating the loss, you can use torch.unbind to get all feature maps can compute the loss
            one by one with the ground truth, then do a weighted average for all losses to achieve the final loss.
        deep_supr_num: number of feature maps that will output during deep supervision head. The
            value should be larger than 0 and less than the number of up sample layers.
            Defaults to 1.
        res_block: whether to use residual connection based convolution blocks during the network.
            Defaults to ``False``.
        trans_bias: whether to set the bias parameter in transposed convolution layers. Defaults to ``False``.
    """
    return EncDecDynUNet(
        DynUNet(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            deep_supervision=deep_supervision, 
            deep_supr_num=deep_supr_num,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block,
            trans_bias=trans_bias
        )
    )


class EncDynUNet(nn.Module):
    def __init__(
        self,
        dynunet:DynUNet
    ) -> None:
        """
            Re-implementation of monai.networks.nets.DynUNet.
            The original implementation constructs the model in a recursive manner, not exposing
            necessary encoder and decoder APIs.
            For testing, the class is initialize with a DynUNet block, while aiming to achieve the same
            behavior as the initialization block without calling its `forward()` function.
        """
        super().__init__()

        def _parse_skip_layers(skip_layers:DynUNetSkipLayer):
            super_head, downsample = nn.ModuleList([]), nn.ModuleList([])
            cur_skip_layer = skip_layers
            while hasattr(cur_skip_layer, "next_layer"):
                super_head.append(cur_skip_layer.super_head)
                downsample.append(cur_skip_layer.downsample)
                # upsample.append(cur_skip_layer.upsample)
                cur_skip_layer = cur_skip_layer.next_layer
            return super_head, downsample, cur_skip_layer
        
        self.filters = dynunet.filters

        self.super_heads, self.downsamples, self.bottleneck = _parse_skip_layers(dynunet.skip_layers)
        self.heads, self.deep_supervision = dynunet.heads, dynunet.deep_supervision

    def down_op(self, x):
        downouts:List[torch.Tensor] = []
        for downsample in self.downsamples:
            x_down = downsample(x)
            downouts.append(x_down)
            x = x_down
        return downouts

    def forward(self, x):
        # downsample
        downouts = self.down_op(x)

        # bottleneck
        bottleneckout = self.bottleneck(downouts[-1])
        # encoder: self.down_op and self.bottleneck

        return bottleneckout

class CNN_UNet(EncDecDynUNet):
    def __init__(
        self,
        mod_channel:int,
        num_class:int,
        dynunet:DynUNet
    ) -> None:
        super().__init__(dynunet)
        self.conv_mod_channel = nn.Conv3d(in_channels=mod_channel, out_channels=32, kernel_size=1, stride=1, padding=0)
        # self.bn=nn.BatchNorm3d(32) # out_channels
        self.IN=nn.InstanceNorm3d(num_features=32)  # out_channels
        # self.GN=nn.GroupNorm(num_class, 32)

    def forward(self, x):
        # downsampling the MR data sample from 4 channels into 1 channel 
        x = self.conv_mod_channel(x)
        x = self.IN(x)
        return super().forward(x)

class CNN_Enc_Dyunet(EncDynUNet):
    def __init__(self,mod_channel,dyunet):
        super().__init__(dyunet)
        self.conv = nn.Conv3d(in_channels=mod_channel, out_channels=32, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(320 * 1, 2)  
        self.dropout = nn.Dropout(0) 

    def forward(self, x):
        x = self.conv(x)
        x= super().forward(x) 
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc(x)
        return x
    
class DUNet_LSE(CNN_UNet):

    def __init__(self,mod_channel,out_channels,dyunet):
        super().__init__(mod_channel,out_channels,dyunet)
        self.r=0.5

    def cal_lse(self, x):
        # drop the background channel
        patch_shape=torch.prod(torch.tensor(x.shape[2:]))
        # logger.info(f"{torch.max(x.reshape(*x.shape[:2], -1), dim=-1)}")
        # logger.info(patch_shape.shape)
        # exit()
        x=torch.sum(torch.exp(self.r*x), dim=(2, 3, 4))/patch_shape
        x=torch.log(x)/self.r
        # x=torch.sigmoid(x)
        return x

    def forward(self, x):
        x=super().forward(x)
        # dimension reduction 
        x = self.cal_lse(x)
        return x


def create_unet_encoder(
    in_channels:int,
    out_channels:int,
    spatial_dims:int=3,
    kernel_size:Sequence[Union[Sequence[int], int]]=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides:Sequence[Union[Sequence[int], int]]=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size:Sequence[Union[Sequence[int], int]]=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters:Optional[Sequence[int]]=None,
    dropout:Optional[Union[Tuple, str, float]] = None,
    norm_name:Union[Tuple, str]=("INSTANCE", {"affine": True}),
    act_name:Union[Tuple, str]=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    deep_supervision:bool=False,
    deep_supr_num:int=3,
    res_block:bool=False,
    trans_bias:bool=False):

    return EncDynUNet(
        DynUNet(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            deep_supervision=deep_supervision, 
            deep_supr_num=deep_supr_num,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block,
            trans_bias=trans_bias
        )
    )


def create_cnn_dynunet(
    mod_channel:int,
    in_channels:int=32,
    out_channels:int=1,
    spatial_dims:int=3,
    kernel_size:Sequence[Union[Sequence[int], int]]=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides:Sequence[Union[Sequence[int], int]]=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size:Sequence[Union[Sequence[int], int]]=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters:Optional[Sequence[int]]=None,
    dropout:Optional[Union[Tuple, str, float]] = None,
    norm_name:Union[Tuple, str]=("INSTANCE", {"affine": True}),
    act_name:Union[Tuple, str]=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    deep_supervision:bool=False,
    deep_supr_num:int=3,
    res_block:bool=False,
    trans_bias:bool=False,
):

    return CNN_UNet(
        mod_channel,
        num_class=out_channels,
        dynunet=DynUNet(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            deep_supervision=deep_supervision, 
            deep_supr_num=deep_supr_num,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block,
            trans_bias=trans_bias
        )
    )

def dyunet_lse(
    mod_channel:int,
    in_channels:int=32,
    out_channels:int=1,
    spatial_dims:int=3,
    kernel_size:Sequence[Union[Sequence[int], int]]=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides:Sequence[Union[Sequence[int], int]]=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size:Sequence[Union[Sequence[int], int]]=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters:Optional[Sequence[int]]=None,
    dropout:Optional[Union[Tuple, str, float]] = None,
    norm_name:Union[Tuple, str]=("INSTANCE", {"affine": True}),
    act_name:Union[Tuple, str]=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    deep_supervision:bool=False,
    deep_supr_num:int=3,
    res_block:bool=False,
    trans_bias:bool=False,
):

    return DUNet_LSE(
    # return CNN_Enc_Dyunet(
        mod_channel,
        out_channels,
        DynUNet(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            deep_supervision=deep_supervision, 
            deep_supr_num=deep_supr_num,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block,
            trans_bias=trans_bias
        )
    )


def dyunet_real_lse(
    mod_channel:int,
    in_channels:int=32,
    out_channels:int=1,
    spatial_dims:int=3,
    kernel_size:Sequence[Union[Sequence[int], int]]=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides:Sequence[Union[Sequence[int], int]]=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size:Sequence[Union[Sequence[int], int]]=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    filters:Optional[Sequence[int]]=None,
    dropout:Optional[Union[Tuple, str, float]] = None,
    norm_name:Union[Tuple, str]=("INSTANCE", {"affine": True}),
    act_name:Union[Tuple, str]=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
    deep_supervision:bool=False,
    deep_supr_num:int=3,
    res_block:bool=False,
    trans_bias:bool=False,
):

    return DUNet_LSE(
    # return CNN_Enc_Dyunet(
        mod_channel,
        DynUNet(
            spatial_dims=spatial_dims, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            deep_supervision=deep_supervision, 
            deep_supr_num=deep_supr_num,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block,
            trans_bias=trans_bias
        )
    )