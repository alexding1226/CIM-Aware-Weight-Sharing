from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

def compute_column_distances_r1(A, B, dist_type="euclidean"):
    if A.shape != B.shape:
        raise ValueError("The shapes of A and B must be the same.")

    # Compute the squared difference
    # Sum along rows and take the square root
    if dist_type == "euclidean":
        distances = torch.sqrt(torch.sum((A - B) ** 2, axis=0))
    elif dist_type == "cosine":
        A_norm = torch.norm(A, dim=0)
        B_norm = torch.norm(B, dim=0)
        distances = 1 - torch.sum(A * B, axis=0) / (A_norm * B_norm)
    elif dist_type == "manhattan":
        distances = torch.sum(torch.abs(A - B), axis=0)
    elif dist_type == "l3norm":
        distances = torch.pow(torch.sum(torch.abs((A - B)**3), axis=0), 0.33333333333333333333333)
    elif dist_type == "l4norm":
        distances = torch.pow(torch.sum(torch.abs((A - B)**4), axis=0), 0.25)
    elif dist_type == "linf":
        distances = torch.max(torch.abs(A - B), axis=0).values


    return distances

def _4D_to_2D(Conv2D_weight):
    original_weight = Conv2D_weight.clone()
    out_channels, in_channels, k_h, k_w = Conv2D_weight.shape
    share_height = in_channels * k_h * k_w
    
    weight = original_weight.view(out_channels, share_height)
    assert weight.ndim == 2
    return weight.clone()

def _2D_to_4D(Conv2D_new_weight, out_channels, in_channels, k_h, k_w):
    new_weight = Conv2D_new_weight.clone()
    share_height = in_channels * k_h * k_w

    weight = new_weight.view(out_channels, in_channels, k_h, k_w)
    assert weight.ndim == 4
    return weight.clone()

def compute_distances_inside_matrix(matrix, mask=None, macro_width=64, macro_height: int = 64, flow: str = "row", dist_type: str ="euclidean", is_conv: bool = True, accumulate_dist_method: str = "mean", test_mode: bool = False, real_macro_height: int = 64):

    if is_conv: matrix = _4D_to_2D(matrix)
    
    width, height = matrix.shape
    
    upd_time_col = height // macro_height
    upd_time_row = width // macro_width

    if len(mask) == 0: return 0
    if macro_height < real_macro_height: return 0
    if upd_time_row*upd_time_col-1 <= 0:    return 0

    def accumulate_dist_function(dist_list, accumulate_dist_method):
        if accumulate_dist_method == "mean":    return torch.mean(dist_list)
        # elif accumulate_dist_method == "sum":   return torch.sum(dist_list)
        else:                                   raise NotImplementedError


    dist_list = []
    if flow == "row":
        for upd in range(upd_time_row*upd_time_col-1):
            upd_col = upd // upd_time_row
            upd_row = upd % upd_time_row
            if upd_row == upd_time_row-1:
                if len(mask) == 0:
                    return torch.zeros(1)
                # dist = torch.mean(compute_column_distances_r1(matrix[upd_row*macro_width:(upd_row+1)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height],matrix[0:macro_width,(upd_col+1)*macro_height:(upd_col+2)*macro_height], dist_type=dist_type) * mask[upd])
                dist = accumulate_dist_function(
                    compute_column_distances_r1(matrix[upd_row*macro_width:(upd_row+1)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height],matrix[0:macro_width,(upd_col+1)*macro_height:(upd_col+2)*macro_height], dist_type=dist_type) * mask[upd],
                    accumulate_dist_method
                )
            else:
                tmp1 = matrix[ upd_row   *macro_width:(upd_row+1)*macro_width, upd_col*macro_height:(upd_col+1)*macro_height]
                tmp2 = matrix[(upd_row+1)*macro_width:(upd_row+2)*macro_width, upd_col*macro_height:(upd_col+1)*macro_height]
                tmp = compute_column_distances_r1(tmp1, tmp2, dist_type=dist_type)
                
                

                if test_mode:
                    print(len(mask[upd]))
                    for tmp_idx, m in enumerate(mask[upd]):
                        if m:
                            print(tmp1[:, tmp_idx])
                            print(tmp2[:, tmp_idx])
                            print(tmp[tmp_idx])
                        else:
                            print("\t", tmp1[:, tmp_idx])
                            print("\t", tmp2[:, tmp_idx])
                            print("\t", tmp[tmp_idx])
                    print("="*20)


                masked_dist = tmp * mask[upd]
                dist = accumulate_dist_function(
                    masked_dist, 
                    accumulate_dist_method
                )
                if test_mode:
                    if not all(y==0 for y in mask[upd]):
                        for mxi, mx in enumerate(mask[upd]):
                            if not mx:
                                print(mx, tmp[mxi])

            
            dist_list.append(dist)
    elif flow == "column":
        for upd in range(upd_time_row*upd_time_col-1):
            upd_col = upd % upd_time_col
            upd_row = upd // upd_time_col
            if upd_col == upd_time_col-1:
                dist = torch.mean(compute_column_distances_r1(matrix[upd_row*macro_width:(upd_row+1)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height],matrix[(upd_row+1)*macro_width:(upd_row+2)*macro_width,0:macro_height], dist_type=dist_type) * mask[upd])
            else:
                dist = torch.mean(compute_column_distances_r1(matrix[upd_row*macro_width:(upd_row+1)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height],matrix[upd_row*macro_width:(upd_row+1)*macro_width,(upd_col+1)*macro_height:(upd_col+2)*macro_height], dist_type=dist_type) * mask[upd])
            dist_list.append(dist)
    else:
        raise ValueError("The flow must be either row or column")

    if accumulate_dist_method == "sum":
        return sum(dist_list)

  

    return sum(dist_list)/len(dist_list)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as Resnet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Resnet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


        #########################################################
        self.num_classes = num_classes

        self.conv_mask = None
        self.fc_mask = None
        self.flow = None
        self.dist_type = None
        self.macro_width = None
        self.macro_height = None
        self.share_height_type = None
        #########################################################


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        # dist = torch.zeros(1).to(x.device)

        dist_conv = self.compute_dist_conv(self.conv_mask) if self.conv_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        dist_fc = self.compute_dist_fc(self.fc_mask) if self.fc_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        dist = dist_conv + dist_fc

        return self._forward_impl(x), dist
    
    def set_mask( self, conv_mask=None, fc_mask=None, flow="row", macro_width=64, macro_height=64, dist_type="euclidean", share_height_type = "macro"):
        self.conv_mask = conv_mask
        self.fc_mask = fc_mask
        self.flow = flow
        self.dist_type = dist_type
        self.macro_width = macro_width
        self.macro_height = macro_height
        self.share_height_type = share_height_type

    def compute_dist_conv(self, conv_mask):
        dist_list = []
        idx = 0
        for conv_layer in self.modules():
            if isinstance(conv_layer, nn.Conv2d):
                weight = conv_layer.weight
                out_channels, in_channels, k_h, k_w = weight.shape

                share_height = in_channels*k_w*k_h
                layer_dist = compute_distances_inside_matrix(weight, conv_mask[idx], self.macro_width, share_height, self.flow, self.dist_type, True, "mean", False, self.macro_height)
                
                dist_list.append(layer_dist)
                idx += 1
        
        return sum(dist_list)

    def compute_dist_fc(self, fc_mask):
        dist_list = [] 
        idx = 0
        for linear_layer in self.modules():
            if isinstance(linear_layer, nn.Linear):
                weight = linear_layer.weight
                w, h = weight.shape
                share_height = h
                layer_dist = compute_distances_inside_matrix(weight, fc_mask[idx], self.macro_width, share_height, self.flow, self.dist_type, False, "mean", False, self.macro_height)
                dist_list.append(layer_dist)
                idx += 1

        return sum(dist_list)



def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> Resnet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = Resnet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def resnet18(**kwargs: Any) -> Resnet:
    return _resnet(BasicBlock, [2, 2, 2, 2], None, True, **kwargs)

def resnet34(**kwargs: Any) -> Resnet:
    return _resnet(BasicBlock, [3, 4, 6, 3], None, True, **kwargs)

def resnet50(**kwargs: Any) -> Resnet:
    return _resnet(Bottleneck, [3, 4, 6, 3], None, True, **kwargs)

