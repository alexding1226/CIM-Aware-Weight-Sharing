
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.registry import register_model

from typing import Callable, List, Optional, Sequence, Tuple, Type, Union
from torch.jit import Final

from timm.models.layers import DropPath, trunc_normal_,PatchEmbed, Mlp

from functools import partial

def compute_column_distances(A, B):
    # Ensure A and B are 2D tensors of shape (N, N)
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape

    # # Step 1: Broadcasting - A is (N, N, 1), B is (N, 1, N)
    # A_expanded = A.unsqueeze(2)  # Shape becomes (N, N, 1)
    # B_expanded = B.unsqueeze(1)  # Shape becomes (N, 1, N)

    # # Step 2: Vectorized difference
    # diff = A_expanded - B_expanded  # Shape becomes (N, N, N)

    distances = torch.sqrt(torch.sum((A.unsqueeze(2) - B.unsqueeze(1))**2, dim=0))
    return distances

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

def compute_distances_inside_matrix(matrix, mask=None, macro_width=64, macro_height=768, flow = "row", dist_type="euclidean"):
    # Ensure matrix is a 2D tensor of shape (N, D)
    assert matrix.ndim == 2

    width, height = matrix.shape
    upd_time_col = height // macro_height
    upd_time_row = width // macro_width

    dist_list = []

    if flow == "row":
        for upd in range(upd_time_row*upd_time_col-1):
            upd_col = upd // upd_time_row
            upd_row = upd % upd_time_row
            if upd_row == upd_time_row-1:
                dist = torch.mean(compute_column_distances_r1(matrix[upd_row*macro_width:(upd_row+1)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height],matrix[0:macro_width,(upd_col+1)*macro_height:(upd_col+2)*macro_height], dist_type=dist_type) * mask[upd])
            else:
                dist = torch.mean(compute_column_distances_r1(matrix[upd_row*macro_width:(upd_row+1)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height],matrix[(upd_row+1)*macro_width:(upd_row+2)*macro_width,upd_col*macro_height:(upd_col+1)*macro_height], dist_type=dist_type) * mask[upd])
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
    return sum(dist_list)/len(dist_list)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class AttentionDS(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_tmp = self.qkv(x)  # (3, B_, num_heads, N, D)
        qkv = qkv_tmp.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        qkv_out = qkv_tmp.reshape(B, N, 3, C).permute(2, 0, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, (qkv_out[0], qkv_out[1], qkv_out[2])


class BlockDS(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionDS(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()


        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.dim = dim


    def forward(self, x):
        
        #x = x + self.ls1(self.attn(self.norm1(x)))
        attn_out, qkv_tuple = self.attn(self.norm1(x))
        x = x + self.ls1(attn_out)
        hidden_out = self.mlp(self.norm2(x))
        x = x + self.ls2(hidden_out)
        return x, qkv_tuple, hidden_out
    
    def compute_dist_qkv(self,qkv_mask,macro_width=64,macro_height=768,flow="row",dist_type="euclidean"):
        num_heads = self.attn.num_heads
        head_dim = self.dim // num_heads
        dist_list = []

        q_weight = self.attn.qkv.weight[:self.dim,:]
        k_weight = self.attn.qkv.weight[self.dim:2*self.dim,:]
        v_weight = self.attn.qkv.weight[2*self.dim:,:]

        q_dist = compute_distances_inside_matrix(q_weight, qkv_mask[0], macro_width, macro_height, flow, dist_type)
        k_dist = compute_distances_inside_matrix(k_weight, qkv_mask[1], macro_width, macro_height, flow, dist_type)
        v_dist = compute_distances_inside_matrix(v_weight, qkv_mask[2], macro_width, macro_height, flow, dist_type)
        
        # for head_idx in range(num_heads-1):
           
        #     q_dist = torch.mean(compute_column_distances_r1(q_weight[head_idx*head_dim:(head_idx+1)*head_dim,:],q_weight[(head_idx+1)*head_dim:(head_idx+2)*head_dim,:])* qkv_mask[0][head_idx])
        #     k_dist = torch.mean(compute_column_distances_r1(k_weight[head_idx*head_dim:(head_idx+1)*head_dim,:],k_weight[(head_idx+1)*head_dim:(head_idx+2)*head_dim,:])* qkv_mask[1][head_idx])
        #     v_dist = torch.mean(compute_column_distances_r1(v_weight[head_idx*head_dim:(head_idx+1)*head_dim,:],v_weight[(head_idx+1)*head_dim:(head_idx+2)*head_dim,:])* qkv_mask[2][head_idx])
        #     dist = (q_dist + k_dist + v_dist)/3
            
        #     # dist = torch.sum(compute_column_distances(self.attn.qkv.weight[head_idx*head_dim:(head_idx+1)*head_dim,:],self.attn.qkv.weight[(head_idx+1)*head_dim:(head_idx+2)*head_dim,:])* qkv_mask[0][head_idx])\
        #     # + torch.sum(compute_column_distances(self.attn.qkv.weight[self.dim+head_idx*head_dim:self.dim+(head_idx+1)*head_dim,:],self.attn.qkv.weight[self.dim+(head_idx+1)*head_dim:self.dim+(head_idx+2)*head_dim,:])* qkv_mask[1][head_idx])\
        #     # + torch.sum(compute_column_distances(self.attn.qkv.weight[2*self.dim+head_idx*head_dim: 2*self.dim+ (head_idx+1)*head_dim,:],self.attn.qkv.weight[2*self.dim+(head_idx+1)*head_dim: 2*self.dim+ (head_idx+2)*head_dim,:])* qkv_mask[2][head_idx])
        #     dist_list.append(dist)
        return (q_dist + k_dist + v_dist)/3
    def compute_dist_fc1(self,fc1_mask,macro_width=64,macro_height=768,flow="row",dist_type="euclidean"):
        weight = self.mlp.fc1.weight
        dist = compute_distances_inside_matrix(weight, fc1_mask, macro_width, macro_height, flow, dist_type)
        return dist
        
    def compute_dist_fc2(self,fc2_mask,macro_width=64,macro_height=768,flow="row",dist_type="euclidean"):
        weight = self.mlp.fc2.weight
        dist = compute_distances_inside_matrix(weight, fc2_mask, macro_width, macro_height, flow, dist_type)
        return dist

class RSVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            no_embed_class=False,
            pre_norm=False,
            fc_norm=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            block_fn=BlockDS,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # self.blocks = nn.Sequential(*[
        #     block_fn(
        #         dim=embed_dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         init_values=init_values,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[i],
        #         norm_layer=norm_layer,
        #         act_layer=act_layer
        #     )
        #     for i in range(depth)])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            ))
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.qkv_mask = None
        self.fc1_mask = None
        self.fc2_mask = None

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}


    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.norm_pre(x)
        #x = self.blocks(x)
        qkv_list = []
        hidden_list = []
        for i in range(len(self.blocks)):
            x, qkv_tuple, hidden_tuple = self.blocks[i](x)
            qkv_list.append(qkv_tuple)
            hidden_list.append(hidden_tuple)
        x = self.norm(x)
        return x, qkv_list, hidden_list

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)
    
    def set_mask(self,qkv_mask,fc1_mask=None,fc2_mask=None,flow="row",macro_width=64,macro_height=768,dist_type="euclidean", share_height_type = "macro"):
        self.qkv_mask = qkv_mask
        self.fc1_mask = fc1_mask
        self.fc2_mask = fc2_mask
        self.flow = flow
        self.dist_type = dist_type
        self.macro_width = macro_width
        self.macro_height = macro_height
        self.share_height_type = share_height_type


    def forward(self, x):
        x, qkv_list, hidden_list = self.forward_features(x)
        x = self.forward_head(x)
        dist_qkv = self.compute_dist_qkv(self.qkv_mask) if self.qkv_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        dist_fc1 = self.compute_dist_fc1(self.fc1_mask) if self.fc1_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        dist_fc2 = self.compute_dist_fc2(self.fc2_mask) if self.fc2_mask is not None else torch.zeros(1,dtype=torch.float32,device=x.device)
        # print(dist_qkv.device,dist_fc1.device,dist_fc2.device)
        dist = dist_qkv + dist_fc1 + dist_fc2
        return x, qkv_list, hidden_list, dist
    
    def compute_dist_qkv(self,qkv_mask):
        dist_list = []
        if self.share_height_type == "macro":
            share_height = self.macro_height
        else:
            share_height = 768
        for idx, block in enumerate(self.blocks):
            block_dist = block.compute_dist_qkv(qkv_mask[idx],self.macro_width,share_height,self.flow,self.dist_type)
            dist_list.append(block_dist)
        # block_dist = self.blocks[0].compute_dist_qkv(qkv_mask[0])
        # dist_list.append(block_dist)
        # block_dist = self.blocks[1].compute_dist_qkv(qkv_mask[1])
        # dist_list.append(block_dist)
        return sum(dist_list)
        

    def compute_dist_fc1(self,fc1_mask):
        dist_list = []
        if self.share_height_type == "macro":
            share_height = self.macro_height
        else:
            share_height = 768
        for idx,block in enumerate(self.blocks):
            block_dist = block.compute_dist_fc1(fc1_mask[idx],self.macro_width,share_height,self.flow,self.dist_type)
            dist_list.append(block_dist)
        return sum(dist_list)

    def compute_dist_fc2(self,fc2_mask):
        dist_list = []
        if self.share_height_type == "macro":
            share_height = self.macro_height
        else:
            share_height = 3072
        for idx,block in enumerate(self.blocks):
            block_dist = block.compute_dist_fc2(fc2_mask[idx],self.macro_width,share_height,self.flow,self.dist_type)
            dist_list.append(block_dist)
        return sum(dist_list)


    @torch.no_grad()
    def copy_weights(self,model_name = "deit3_base_patch16_224"):
        print("model_name : ",model_name)
        pretrained_model = timm.create_model(model_name, pretrained=True)

        self.cls_token.copy_(pretrained_model.cls_token)

        self.patch_embed.proj.weight.copy_(pretrained_model.patch_embed.proj.weight)
        self.patch_embed.proj.bias.copy_(pretrained_model.patch_embed.proj.bias)
        
        self.pos_embed.copy_(pretrained_model.pos_embed)

        for idx, block in enumerate(self.blocks):
            block.norm1.weight.copy_(pretrained_model.blocks[idx].norm1.weight)
            block.norm1.bias.copy_(pretrained_model.blocks[idx].norm1.bias)

            block.attn.qkv.weight.copy_(pretrained_model.blocks[idx].attn.qkv.weight)
            block.attn.qkv.bias.copy_(pretrained_model.blocks[idx].attn.qkv.bias)

            block.attn.proj.weight.copy_(pretrained_model.blocks[idx].attn.proj.weight)
            block.attn.proj.bias.copy_(pretrained_model.blocks[idx].attn.proj.bias)

            block.norm2.weight.copy_(pretrained_model.blocks[idx].norm2.weight)
            block.norm2.bias.copy_(pretrained_model.blocks[idx].norm2.bias)

            block.mlp.fc1.weight.copy_(pretrained_model.blocks[idx].mlp.fc1.weight)
            block.mlp.fc1.bias.copy_(pretrained_model.blocks[idx].mlp.fc1.bias)
            
            block.mlp.fc2.weight.copy_(pretrained_model.blocks[idx].mlp.fc2.weight)
            block.mlp.fc2.bias.copy_(pretrained_model.blocks[idx].mlp.fc2.bias)
            try:
                block.ls1.gamma.copy_(pretrained_model.blocks[idx].ls1.gamma)
                block.ls2.gamma.copy_(pretrained_model.blocks[idx].ls2.gamma)
            except:
                print("no gamma")

        self.norm.weight.copy_(pretrained_model.norm.weight)
        self.norm.bias.copy_(pretrained_model.norm.bias)
    
        self.head.weight.copy_(pretrained_model.head.weight)
        self.head.bias.copy_(pretrained_model.head.bias)


    

    
    

    def freeze_parameters(self):
        for params in self.parameters():
            params.requires_grad = False
    
    def unfreeze_qkv(self):
        for idx, block in enumerate(self.blocks):
            block.attn.qkv.weight.requires_grad = True

    def check_weight(self,idx_mapping_list):
        head_num = 12
        head_dim = self.embed_dim // head_num
        
        for idx, block in enumerate(self.blocks):
            q_weights = block.attn.qkv.weight.data[:self.embed_dim,:]
            k_weights = block.attn.qkv.weight.data[self.embed_dim:2*self.embed_dim,:]
            v_weights = block.attn.qkv.weight.data[2*self.embed_dim:,:]
            q_share_idx = idx_mapping_list[idx][0]
            k_share_idx = idx_mapping_list[idx][1]
            v_share_idx = idx_mapping_list[idx][2]

            for share_idx in q_share_idx:
                start_head = share_idx[0]
                nxt_head = start_head+1
                weight_origin = q_weights[start_head*head_dim : (start_head+1)*head_dim, share_idx[1]]

                for id in share_idx[2:]:
                    assert torch.equal(weight_origin, q_weights[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    nxt_head = nxt_head+1
                

            for share_idx in k_share_idx:
                start_head = share_idx[0]
                nxt_head = start_head+1
                weight_origin = k_weights[start_head*head_dim : (start_head+1)*head_dim, share_idx[1]]

                for id in share_idx[2:]:
                    assert torch.equal(weight_origin, k_weights[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    nxt_head = nxt_head+1

            
            for share_idx in v_share_idx:
                start_head = share_idx[0]
                nxt_head = start_head+1
                weight_origin = v_weights[start_head*head_dim : (start_head+1)*head_dim, share_idx[1]]

                for id in share_idx[2:]:
                    assert torch.equal(weight_origin, v_weights[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    nxt_head = nxt_head+1

    def check_grad(model,idx_mapping_list):

        head_num = 12
        head_dim = model.embed_dim // head_num
        for idx, block in enumerate(model.blocks):
            q_grads = block.attn.qkv.weight.grad[:model.embed_dim,:]
            k_grads = block.attn.qkv.weight.grad[model.embed_dim:2*model.embed_dim,:]
            v_grads = block.attn.qkv.weight.grad[2*model.embed_dim:,:]

            q_share_idx = idx_mapping_list[idx][0]
            k_share_idx = idx_mapping_list[idx][1]
            v_share_idx = idx_mapping_list[idx][2]

            for share_idx in q_share_idx:
                start_head = share_idx[0]
                grad = q_grads[start_head*head_dim : (start_head+1)*head_dim, share_idx[1]]
                nxt_head = start_head+1

                for id in share_idx[2:]:

                    if not torch.equal(grad, q_grads[nxt_head*head_dim :(nxt_head+1)*head_dim, id]):
                        print("grad error")
                        print("share_idx : ",share_idx)
                        print("id : ",id)
                        print(grad)
                        print(q_grads[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    assert torch.equal(grad, q_grads[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    nxt_head = nxt_head+1
                

            for share_idx in k_share_idx:
                start_head = share_idx[0]
                grad = k_grads[start_head*head_dim : (start_head+1)*head_dim, share_idx[1]]
                nxt_head = start_head+1

                for id in share_idx[2:]:
                    assert torch.equal(grad, k_grads[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    nxt_head = nxt_head+1

            
            for share_idx in v_share_idx:
                start_head = share_idx[0]
                grad = v_grads[start_head*head_dim : (start_head+1)*head_dim, share_idx[1]]
                nxt_head = start_head+1

                for id in share_idx[2:]:
                    assert torch.equal(grad, v_grads[nxt_head*head_dim :(nxt_head+1)*head_dim, id])
                    nxt_head = nxt_head+1


                
