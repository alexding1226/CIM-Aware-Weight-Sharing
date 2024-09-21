# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import cast, Dict, List, Union

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "VGG",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

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
    original_weight = Conv2D_weight.clone().detach()
    out_channels, in_channels, k_h, k_w = Conv2D_weight.shape
    share_height = in_channels * k_h * k_w
    
    weight = original_weight.view(out_channels, share_height)
    assert weight.ndim == 2
    return weight.clone().detach()

def _2D_to_4D(Conv2D_new_weight, out_channels, in_channels, k_h, k_w):
    new_weight = Conv2D_new_weight.clone().detach()
    share_height = in_channels * k_h * k_w

    weight = new_weight.view(out_channels, in_channels, k_h, k_w)
    assert weight.ndim == 4
    return weight.clone().detach()

def pad_logic():
    return """
    if (out_channels < macro_height) or (in_channels*k_h*k_w < macro_width):
        pad_width = (0, max(0, macro_height - out_channels))  # Padding for width (height dimension in 2D)
        pad_height = (0, max(0, macro_width - in_channels*k_h*k_w))  # Padding for height (width dimension in 2D)

        # Pad the weight matrix using torch.nn.functional.pad
        matrix = nn.functional.pad(matrix, pad=pad_width + pad_height, mode='constant', value=0)
    """

"""
# Ensure upd_time_row and upd_time_col are at least 1
    if upd_time_row == 0 or upd_time_col == 0:
        pad_width = (0, max(0, macro_height - height))  # Padding for width (height dimension in 2D)
        pad_height = (0, max(0, macro_width - width))  # Padding for height (width dimension in 2D)
        
        # Pad the weight matrix using torch.nn.functional.pad
        matrix = nn.functional.pad(matrix, pad=pad_width + pad_height, mode='constant', value=0)
        
        # Update mat_width and mat_height after padding
        mat_width, mat_height = matrix.shape
        upd_time_row = max(1, mat_width // macro_width)  # Ensure at least 1
        upd_time_col = max(1, mat_height // macro_height)  # Ensure at least 1
"""

def compute_distances_inside_matrix(matrix, mask=None, macro_width=64, macro_height: int = 64, flow: str = "row", dist_type: str ="euclidean", is_conv: bool = True, accumulate_dist_method: str = "mean", test_mode: bool = False):

    if is_conv: matrix = _4D_to_2D(matrix)
    
    width, height = matrix.shape
    
    upd_time_col = height // macro_height
    upd_time_row = width // macro_width


    if upd_time_row*upd_time_col-1 <= 0:    return 0

    def accumulate_dist_function(dist_list, accumulate_dist_method):
        if accumulate_dist_method == "mean":    return torch.mean(dist_list)
        elif accumulate_dist_method == "sum":   return torch.sum(dist_list)
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
                tmp1 = matrix[ upd_row   *macro_width:(upd_row+1)*macro_width, upd_col*macro_height:(upd_col+1)*macro_height].clone().detach()
                tmp2 = matrix[(upd_row+1)*macro_width:(upd_row+2)*macro_width, upd_col*macro_height:(upd_col+1)*macro_height].clone().detach()
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



def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class VGG(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, num_classes: int = 1000) -> None:
        super(VGG, self).__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        # Initialize neural network weights
        self._initialize_weights()

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


    def forward(self, x: Tensor) -> Tensor:
        x = self._forward_impl(x)
        dist_conv = self.compute_dist_conv(self.conv_mask) if self.conv_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        dist_fc = self.compute_dist_fc(self.fc_mask) if self.fc_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        dist = dist_conv + dist_fc
        
        """ # Pony's test
        print_dist_conv, print_dist_fc, print_dist = dist_conv, dist_fc, dist
        if self.conv_mask is None:
            print_dist_conv = None
        else:
            if len(self.conv_mask) == 0:
                print("Empty conv mask")

        if self.fc_mask is None:
            print_dist_fc = None        
        else:
            if len(self.fc_mask) == 0:
                print("Empty fc mask")    
        print(f"dist_conv: {print_dist_conv}, dist_fc: {print_dist_fc}, dist: {print_dist}")
        """

        # dist = dist_conv
        # dist = torch.zeros(1,dtype=torch.float32,device=x.device)
        return x, dist
    
    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    @torch.no_grad()
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

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
        for i, conv_layer in enumerate(self.features):
            if isinstance(conv_layer, nn.Conv2d):
                weight = conv_layer.weight
                out_channels, in_channels, k_h, k_w = weight.shape

                share_height = in_channels*k_w*k_h
                layer_dist = compute_distances_inside_matrix(weight, conv_mask[idx], self.macro_width, share_height, self.flow, self.dist_type, True)
                
                dist_list.append(layer_dist)
                idx += 1
        
        return sum(dist_list)

    def compute_dist_fc(self, fc_mask):
        dist_list = [] 
        idx = 0
        for linear_layer in self.classifier:
            if isinstance(linear_layer, nn.Linear):
                weight = linear_layer.weight
                w, h = weight.shape
                share_height = h
                layer_dist = compute_distances_inside_matrix(weight, fc_mask[idx], self.macro_width, share_height, self.flow, self.dist_type, False)
                dist_list.append(layer_dist)
                idx += 1

        return sum(dist_list)

    def update_weight(self, layer_type, layer_index, new_weight):
        if layer_type == "conv":
            self.features[layer_index].weight = new_weight
        elif layer_type == "fc":
            self.classifier[layer_index].weight = new_weight

    def unfreeze(self):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = True
        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                layer.weight.requires_grad = True

    def freeze_parameters(self):
        for params in self.parameters():
            params.requires_grad = False


def vgg11(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], False, **kwargs)

    return model


def vgg13(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], False, **kwargs)

    return model


def vgg16(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], False, **kwargs)

    return model


def vgg19(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], False, **kwargs)

    return model


def vgg11_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg11"], True, **kwargs)

    return model


def vgg13_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg13"], True, **kwargs)

    return model


def vgg16_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg16"], True, **kwargs)

    return model


def vgg19_bn(**kwargs) -> VGG:
    model = VGG(vgg_cfgs["vgg19"], True, **kwargs)

    return model