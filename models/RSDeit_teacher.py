
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.registry import register_model

from typing import Callable, List, Optional, Sequence, Tuple, Type, Union
from torch.jit import Final

from timm.models.layers import DropPath, trunc_normal_,PatchEmbed, Mlp

from functools import partial

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

    def forward(self, x):
        
        #x = x + self.ls1(self.attn(self.norm1(x)))
        attn_out, qkv_tuple = self.attn(self.norm1(x))
        x = x + self.ls1(attn_out)
        hidden_out = self.mlp(self.norm2(x))
        x = x + self.ls2(hidden_out)
        return x, qkv_tuple, hidden_out

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

    def forward(self, x):
        x, qkv_list, hidden_list = self.forward_features(x)
        x = self.forward_head(x)
        return x, qkv_list, hidden_list


    @torch.no_grad()
    def copy_weights(self,model_name = "deit3_base_patch16_224"):
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

            block.ls1.gamma.copy_(pretrained_model.blocks[idx].ls1.gamma)
            block.ls2.gamma.copy_(pretrained_model.blocks[idx].ls2.gamma)

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


                
