import time, os
import json, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torchvision import transforms, utils



def compute_column_distances_r1(A, B, dist_type = "euclidean"):
    if A.shape != B.shape:
        raise ValueError("The shapes of A and B must be the same.")
 
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
    else:
        raise ValueError("dist_type must be 'euclidean' or 'cosine' or 'manhattan'.")

    return distances


def pad_weight(weight, height, width):
    pad_width = (0, max(0, height - weight.shape[0]))  # Padding for width (height dimension in 2D)
    pad_height = (0, max(0, width - weight.shape[1]))  # Padding for height (width dimension in 2D)
    new_weight = nn.functional.pad(weight, pad=pad_width + pad_height, mode='constant', value=0)

    """
    # Ensure upd_time_row and upd_time_col are at least 1
    if upd_time_row == 0 or upd_time_col == 0:
        # raise ValueError("Theoritically, upd_time_row and upd_time_col should be at least 1.")
        pad_width = (0, max(0, share_height - mat_height))  # Padding for width (height dimension in 2D)
        pad_height = (0, max(0, macro_width - mat_width))  # Padding for height (width dimension in 2D)
        
        # Pad the weight matrix using torch.nn.functional.pad
        weight = nn.functional.pad(weight, pad=pad_width + pad_height, mode='constant', value=0)
        
        # Update mat_width and mat_height after padding
        mat_width, mat_height = weight.shape
        upd_time_row = mat_width // macro_width
        upd_time_col = mat_height // share_height

        assert upd_time_row > 0 and upd_time_col > 0, "Error: upd_time_row and upd_time_col must be at least 1 after padding"

    # print(f"padded upd_time_row : {upd_time_row}, upd_time_col : {upd_time_col}")    
    """

    return new_weight.clone().detach()

def _4D_to_2D(Conv2D_weight):
    original_weight = Conv2D_weight.clone()
    out_channels, in_channels, k_h, k_w = Conv2D_weight.shape
    share_height = in_channels * k_h * k_w
    
    weight = original_weight.view(out_channels, share_height)
    return weight.clone()

def _2D_to_4D(Conv2D_new_weight, out_channels, in_channels, k_h, k_w):
    new_weight = Conv2D_new_weight.clone()
    share_height = in_channels * k_h * k_w

    weight = new_weight.view(out_channels, in_channels, k_h, k_w)
    return weight.clone()


def test():
    out_channels, in_channels, k_h, k_w = 64, 3, 3, 3
    share_height = in_channels * k_h * k_w
    test_tensor = torch.rand(out_channels, in_channels, k_h, k_w)

    tmp0 = _4D_to_2D(test_tensor.clone().detach())
    tmp1 = _2D_to_4D(tmp0, out_channels, in_channels, k_h, k_w)

    assert torch.equal(tmp1, test_tensor)

    out_channels, in_channels, k_h, k_w = 64, 3, 3, 3
    share_height = in_channels * k_h * k_w
    test_tensor = torch.rand(out_channels, in_channels, k_h, k_w)


if __name__ == "__main__":
    test()

@torch.no_grad()
def row_sharing_vgg(weight, distance_boundary, max_sharing_rate=0.5, return_shared_index=False, macro_height=64, flow="row", no_sharing=False, macro_width=64, dist_type="euclidean", share_height=64, min_sharing_rate_per_macro=0.7, is_conv=False):
    
    original_weight = weight.clone()

    if is_conv: 
        weight = _4D_to_2D(weight)

    mat_width, mat_height = weight.shape # mat_width = out_channels, mat_height = in_channels * k_h * k_w

    
    upd_time_row = mat_width // macro_width
    upd_time_col = mat_height // share_height


    if (upd_time_row == 0 and upd_time_col == 1) or (upd_time_row == 1 and upd_time_col == 0) or (upd_time_row == 1 and upd_time_col == 1):
        return original_weight, 0, [], 0, 1
    
    
    x = pad_weight(weight, mat_height, mat_width)    

    weight = weight.clone().detach()

    head_weight_list = []
    
    if flow == "column":
        for i in range(upd_time_row):
            for j in range(upd_time_col):
                head_weight_list.append(weight[i*macro_width:(i+1)*macro_width, j*share_height:(j+1)*share_height])
    elif flow == "row":
        for i in range(upd_time_col):
            for j in range(upd_time_row):
                head_weight_list.append(weight[j*macro_width:(j+1)*macro_width, i*share_height:(i+1)*share_height])
    else:
        raise Exception("flow should be column or row")

    num_sharing = 0
    num_train = 0

    mask_allhead = []
    mask_allhead_old = []
    mask_diff_list = []

    for upd_time in range(upd_time_row * upd_time_col - 1):
        first_head_weight = head_weight_list[upd_time]
        second_head_weight = head_weight_list[upd_time + 1]
        distances = compute_column_distances_r1(second_head_weight, first_head_weight, dist_type=dist_type)
        distances = distances.detach().to(torch.device("cpu"))
        mask_train = torch.ones(share_height, dtype=torch.bool, device=distances.device)
        mask_share = torch.ones(share_height, dtype=torch.bool, device=distances.device)
        no_share_row_per_macro = torch.zeros(share_height//macro_height, dtype=int, device=distances.device)
        sharing_row = int(share_height * max_sharing_rate)
        num_no_sharing_row = share_height - sharing_row
        max_no_share_row_per_macro = macro_height - int(macro_height * min_sharing_rate_per_macro * max_sharing_rate)

        sort_value, sort_idx = torch.sort(distances, descending=True)

        no_share_row = 0
        i = 0
        while (sort_value[i] > distance_boundary or no_share_row < num_no_sharing_row):
            idx = sort_idx[i]
            macro_idx = idx // macro_height

            if sort_value[i] > distance_boundary:
                mask_share[idx] = False

            if no_share_row_per_macro[macro_idx] < max_no_share_row_per_macro and no_share_row < num_no_sharing_row:
                mask_train[idx] = False
                mask_share[idx] = False
                no_share_row += 1
                no_share_row_per_macro[macro_idx] += 1
            i += 1
            if i == share_height:
                break

        if not no_sharing:
            head_weight_list[upd_time + 1][:, mask_share] = head_weight_list[upd_time][:, mask_share].clone()


        num_sharing += sum(mask_share)
        num_train += sum(mask_train)
        
        mask_allhead.append(mask_train.to(weight.device))
    
        # debug
        first_head_weight = head_weight_list[upd_time].clone()
        second_head_weight = head_weight_list[upd_time+1].clone()
        distances = compute_column_distances_r1(second_head_weight, first_head_weight, dist_type=dist_type)
        distances = distances.detach().to(torch.device("cpu"))
        mask_old = torch.zeros(share_height, dtype=torch.bool, device=distances.device)
        sort_value, sort_idx = torch.sort(distances)

        for i in range(int(share_height*max_sharing_rate)):
            if sort_value[i] < distance_boundary:
                idx = sort_idx[i]
                mask_old[idx] = True
                if not no_sharing:
                    head_weight_list[upd_time+1][:,idx] = head_weight_list[upd_time][:,idx]
        
        mask_allhead_old.append(mask_old.to(weight.device))
        mask_diff = torch.sum(mask_share ^ mask_old)
        mask_diff_list.append(mask_diff)
        

    new_weight = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
    if flow == "column":
        for i in range(upd_time_row):
            for j in range(upd_time_col):
                new_weight[i*macro_width:(i+1)*macro_width,j*share_height:(j+1)*share_height] = head_weight_list[i*upd_time_col+j]
    elif flow == "row":
        for i in range(upd_time_col):
            for j in range(upd_time_row):
                new_weight[j*macro_width:(j+1)*macro_width,i*share_height:(i+1)*share_height] = head_weight_list[i*upd_time_row+j]

    num_sharing = num_sharing / (upd_time_row*upd_time_col-1)
    num_sharing = num_sharing / share_height

    num_train = num_train / (upd_time_row*upd_time_col-1)
    num_train = num_train / share_height

    # Reshape the weight back to the original convolutional layer shape
    if is_conv:
        out_channels, in_channels, k_h, k_w = original_weight.shape
        new_weight = _2D_to_4D(new_weight, out_channels, in_channels, k_h, k_w)
        assert new_weight.shape == original_weight.shape, "Weight reshaping failed"
        # new_weight = new_weight[:out_channels, :in_channels * k_h * k_w]    # Remove extra rows/columns padded earlier
        # new_weight = new_weight.view(out_channels, in_channels, k_h, k_w)   # Reshape to (in_channels, out_channels, k_h, k_w)


    if return_shared_index:
        return new_weight, num_sharing, mask_allhead, sum(mask_diff_list) / len(mask_diff_list), num_train
    else:
        return new_weight, num_sharing

def all_zero(x: list):
    return all(y == 0 for y in x)

def weight_share_vgg(model, conv_ratio_list, fc_ratio_list, no_sharing=False, macro_width=64, args=None, distance_boundary=100.0, set_mask=True):
    print("Start Weight Sharing VGG")

    conv_boundary_list = [distance_boundary] * len(model.features)
    fc_boundary_list = [distance_boundary] * len(model.classifier)
    
    # conv_sharing_rate_list = [conv_ratio] * len(model.features)
    conv_sharing_rate_list = conv_ratio_list
    conv_mask = None

    if args.share_height_type == "macro":
        share_height = args.macro_height
    
    if not all_zero(conv_ratio_list):
        conv_mask = []
        conv_sharing_block_list = []
        conv_train_block_list = []
        conv_idx = 0
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                """
                torch.Size([64, 3, 3, 3])
                torch.Size([64, 64, 3, 3])
                torch.Size([128, 64, 3, 3])
                torch.Size([128, 128, 3, 3])
                torch.Size([256, 128, 3, 3])
                torch.Size([256, 256, 3, 3])
                torch.Size([256, 256, 3, 3])
                torch.Size([512, 256, 3, 3])
                torch.Size([512, 512, 3, 3])
                torch.Size([512, 512, 3, 3])
                torch.Size([512, 512, 3, 3])
                torch.Size([512, 512, 3, 3])
                torch.Size([512, 512, 3, 3])
                """
                weight = layer.weight.data

                if args.share_height_type == "whole":
                    out_channels, in_channels, k_h, k_w = weight.shape  
                    share_height = in_channels*k_h*k_w

                # print("Current Conv2D shape: ", weight.shape)

                new_weight, num_sharing, mask, mask_diff, num_train = row_sharing_vgg(
                    weight, distance_boundary=conv_boundary_list[conv_idx], max_sharing_rate=conv_sharing_rate_list[conv_idx], 
                    return_shared_index=True, macro_height=args.macro_height, flow=args.flow, 
                    no_sharing=no_sharing, macro_width=args.macro_width, dist_type=args.dist_type, 
                    share_height=share_height, min_sharing_rate_per_macro=args.min_sharing_rate_per_macro, is_conv=True
                )

                if not no_sharing:
                    """ # Test different features per layer nums
                    test_num = 0
                    assert model.features[i].weight.shape == new_weight.shape, f"Dimension mismatch: model weight shape {model.features[i].weight.shape} != new weight shape {weight.shape}"
                    for i_x, x in enumerate(new_weight):
                        if not torch.equal(model.features[i].weight[i_x], x):
                            test_num += 1
                    if test_num > 0:
                        for _ in range(100):
                            print(test_num)
                    """

                    model.features[i].weight.data = new_weight.clone()
                    # print("new_weight =", new_weight.shape)
                    
                
                conv_sharing_block_list.append(num_sharing)
                conv_train_block_list.append(num_train)
                conv_mask.append(mask)

                # Start iterating another layer
                conv_idx += 1

        print(f"Conv sharing block list: {conv_sharing_block_list}")
        print(f"Conv train block list: {conv_train_block_list}")


    # Share FC

    # fc_sharing_rate_list = [fc_ratio] * len(model.classifier)
    fc_sharing_rate_list = fc_ratio_list
    fc_mask = None
    

    # if fc_ratio > 0:
    if not all_zero(fc_ratio_list):
        fc_mask = []
        fc_sharing_block_list = []
        fc_train_block_list = []
        fc_idx = 0
        for i, layer in enumerate(model.classifier):
            if isinstance(layer, nn.Linear):
                weight = layer.weight.data
                print("Current FC shape: ", weight.shape)

                if args.share_height_type == "whole":
                    share_height = weight.shape[1]

                weight, num_sharing, mask, mask_diff, num_train = row_sharing_vgg(
                    weight, distance_boundary=fc_boundary_list[fc_idx], max_sharing_rate=fc_sharing_rate_list[fc_idx],
                    return_shared_index=True, macro_height=args.macro_height, flow=args.flow, 
                    no_sharing=no_sharing, macro_width=args.macro_width, dist_type=args.dist_type, 
                    share_height=share_height, min_sharing_rate_per_macro=args.min_sharing_rate_per_macro, is_conv=False
                )
                if not no_sharing:
                    model.classifier[i].weight.data = weight.clone()
                    
                fc_sharing_block_list.append(num_sharing)
                fc_train_block_list.append(num_train)
                fc_mask.append(mask)

                # Start iterating another layer
                fc_idx += 1

        print(f"FC sharing block list: {fc_sharing_block_list}")
        print(f"FC train block list: {fc_train_block_list}")

    if set_mask:
        model.set_mask(conv_mask, fc_mask, flow=args.flow, macro_width=args.macro_width, macro_height=args.macro_height, dist_type=args.dist_type, share_height_type=args.share_height_type)

    print("End Weight Sharing VGG")

    
"""
def compute_column_distances(A, B):
    # Ensure A and B are 2D tensors of shape (N, N)
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape

    # Step 1: Broadcasting - A is (N, N, 1), B is (N, 1, N)
    A_expanded = A.unsqueeze(2)  # Shape becomes (N, N, 1)
    B_expanded = B.unsqueeze(1)  # Shape becomes (N, 1, N)

    # Step 2: Vectorized difference
    diff = A_expanded - B_expanded  # Shape becomes (N, N, N)
    diff = diff**2
    
    # Step 3: Square and sum along the first dimension
    dist_squared = torch.sum(diff, dim=0)

    # Step 4: Square root
    distances = torch.sqrt(dist_squared)

    return distances
def compute_grouped_column_distances(A, B, group_size):
    # Ensure A and B are 2D tensors of shape (N, N)
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape
    N = A.shape[1]
    assert N % group_size == 0  # Ensure N is divisible by the group size

    # Initialize the distance matrix with a large value
    distances = torch.full((N, N), float(1000000), device=A.device)

    # Iterate over the groups
    for i in range(0, N, group_size):
        # Indices of the current group
        group_indices = slice(i, i + group_size)

        # Extract the columns for the current group
        A_group = A[:, group_indices].unsqueeze(2)  # Shape becomes (N, group_size, 1)
        B_group = B[:, group_indices].unsqueeze(1)  # Shape becomes (N, 1, group_size)

        # Compute the distances within the group
        diff = A_group - B_group  # Shape becomes (N, group_size, group_size)
        dist_squared = torch.sum(diff ** 2, dim=0)
        distances_group = torch.sqrt(dist_squared)

        # Assign these distances to the corresponding positions in the distance matrix
        distances[group_indices, group_indices] = distances_group

    return distances

@torch.no_grad()
@torch.no_grad()
def row_sharing_r1(weight, distance_boundary, max_sharing_rate=0.5, return_shared_index = False, macro_height = 64, flow = "row", no_sharing = False, macro_width = 64, dist_type = "euclidean", share_height = 768, min_sharing_rate_per_macro = 0.7):
    print(weight.shape)
    mat_width, mat_height = weight.shape 
    upd_time_row = mat_width // macro_width 
    head_weight_list = []
    upd_time_col = mat_height // share_height
    weight = weight.clone().detach()
    if flow == "column":
        for i in range(upd_time_row):
            for j in range(upd_time_col):
                head_weight_list.append(weight[i*macro_width:(i+1)*macro_width,j*share_height:(j+1)*share_height])
    elif flow == "row":
        for i in range(upd_time_col):
            for j in range(upd_time_row):
                head_weight_list.append(weight[j*macro_width:(j+1)*macro_width,i*share_height:(i+1)*share_height])
    else:
        raise Exception("flow should be column or row")
    # head_weight_list[0].shape # [64, 768]
    # head_weight_list[0][:,0].shape # [64], a row vector of CIM
    num_sharing = 0
    num_train = 0

    mask_allhead = []
    mask_allhead_old = []
    mask_diff_list = []



    for upd_time in range(upd_time_row*upd_time_col-1):
        # if (upd_time % upd_time_row == upd_time_row-1) and (flow == "row"): # the last submat of each row
        #     continue
        
        first_head_weight = head_weight_list[upd_time]
        second_head_weight = head_weight_list[upd_time+1]
        distances = compute_column_distances_r1(second_head_weight, first_head_weight, dist_type=dist_type)
        distances = distances.detach().to(torch.device("cpu"))
        mask_train = torch.ones(share_height, dtype=torch.bool, device=distances.device)
        mask_share = torch.ones(share_height, dtype=torch.bool, device=distances.device)
        no_share_row_per_macro = torch.zeros(share_height//macro_height, dtype=int, device=distances.device)
        sharing_row = int(share_height*max_sharing_rate)
        num_no_sharing_row = share_height - sharing_row
        max_no_share_row_per_macro = macro_height - int(macro_height*min_sharing_rate_per_macro*max_sharing_rate)



        

        # print("distances : ",distances)

        sort_value, sort_idx = torch.sort(distances, descending=True)
        # print("sort_value : ",sort_value)
        # print("sort_idx : ",sort_idx)

        no_share_row = 0
        i = 0
        while (sort_value[i] > distance_boundary or no_share_row < num_no_sharing_row):
            idx = sort_idx[i]
            macro_idx = idx // macro_height

            if sort_value[i] > distance_boundary: # if the distance is larger than the boundary, then it has to not share. but still have a chance to train
                mask_share[idx] = False
            

            if (no_share_row_per_macro[macro_idx] < max_no_share_row_per_macro) and no_share_row < num_no_sharing_row : # if the macro has not reached the maximum number of no sharing row, then it has to not share and not train

                mask_train[idx] = False
                mask_share[idx] = False
                no_share_row += 1
                no_share_row_per_macro[macro_idx] += 1
            i += 1
            if i == share_height: 
                break

        if not no_sharing:
            head_weight_list[upd_time+1][:,mask_share] = head_weight_list[upd_time][:,mask_share]
        

        num_sharing += sum(mask_share)
        num_train += sum(mask_train)

        mask_allhead.append(mask_train.to(weight.device))

        # debug

        first_head_weight = head_weight_list[upd_time]
        second_head_weight = head_weight_list[upd_time+1]
        distances = compute_column_distances_r1(second_head_weight, first_head_weight, dist_type=dist_type)
        distances = distances.detach().to(torch.device("cpu"))
        mask_old = torch.zeros(share_height, dtype=torch.bool, device=distances.device)

        # print("distances : ",distances)

        sort_value, sort_idx = torch.sort(distances)
        # print("sort_value : ",sort_value)
        # print("sort_idx : ",sort_idx)


        for i in range(int(share_height*max_sharing_rate)):
            if sort_value[i] < distance_boundary:
                idx = sort_idx[i]
                mask_old[idx] = True
                if not no_sharing:
                    head_weight_list[upd_time+1][:,idx] = head_weight_list[upd_time][:,idx]


        mask_allhead_old.append(mask_old.to(weight.device))
        mask_diff = torch.sum(mask_share ^ mask_old)
        mask_diff_list.append(mask_diff)
    
    #print("mask diff average : ",sum(mask_diff_list)/len(mask_diff_list))

    new_weight = torch.zeros_like(weight, device=weight.device, dtype=weight.dtype)
    if flow == "column":
        for i in range(upd_time_row):
            for j in range(upd_time_col):
                new_weight[i*macro_width:(i+1)*macro_width,j*share_height:(j+1)*share_height] = head_weight_list[i*upd_time_col+j]
    elif flow == "row":
        for i in range(upd_time_col):
            for j in range(upd_time_row):
                new_weight[j*macro_width:(j+1)*macro_width,i*share_height:(i+1)*share_height] = head_weight_list[i*upd_time_row+j]


    num_sharing = num_sharing / (upd_time_row*upd_time_col-1)
    num_sharing = num_sharing / share_height

    num_train = num_train / (upd_time_row*upd_time_col-1)
    num_train = num_train / share_height

    if return_shared_index:
        return new_weight, num_sharing, mask_allhead, sum(mask_diff_list)/len(mask_diff_list), num_train
    else:
        return new_weight, num_sharing

def determin_boundary(weight, share_height = 768,macro_width = 64, flow = "row", dist_type = "euclidean", boundary_value = -1):
    mat_width, mat_height = weight.shape 
    upd_time_row = mat_width // macro_width 
    head_weight_list = []
    upd_time_col = mat_height // share_height
    weight = weight.clone().detach()
    if flow == "column":
        for i in range(upd_time_row):
            for j in range(upd_time_col):
                head_weight_list.append(weight[i*macro_width:(i+1)*macro_width,j*share_height:(j+1)*share_height])
    elif flow == "row":
        for i in range(upd_time_col):
            for j in range(upd_time_row):
                head_weight_list.append(weight[j*macro_width:(j+1)*macro_width,i*share_height:(i+1)*share_height])
    else:
        raise Exception("flow should be column or row")
    
    for upd_time in range(upd_time_row*upd_time_col-1):
        # if (upd_time % upd_time_row == upd_time_row-1) and (flow == "row"): # the last submat of each row
        #     continue
        
        first_head_weight = head_weight_list[upd_time]
        second_head_weight = head_weight_list[upd_time+1]
        distances = compute_column_distances_r1(second_head_weight, first_head_weight, dist_type=dist_type)
        distances = distances.detach().to(torch.device("cpu"))

        sort_value, sort_idx = torch.sort(distances)

        less_than_boundary = sort_value[sort_value < boundary_value]
        less_than_boundary_ratio = len(less_than_boundary) / len(sort_value)

    return less_than_boundary_ratio
    
def check_distance(model,macro_width=64,args=None, distance_boundary=-1):
    dim = 768
    q_bound = []
    k_bound = []
    v_bound = []
    fc1_bound = []
    fc2_bound = []
    if args.share_height_type == "macro":
        share_height = args.macro_height
        share_height_fc2 = args.macro_height
    elif args.share_height_type == "whole":
        share_height = 768
        share_height_fc2 = 3072
    else:
        raise Exception("share_height_type should be macro or whole")
    for i in range(12):
        weight = model.blocks[i].attn.qkv.weight
        q_weight = weight[:dim,:]
        k_weight = weight[dim:dim*2,:]
        v_weight = weight[dim*2:,:]
        q_boundary = determin_boundary(q_weight, share_height=share_height,macro_width=args.macro_width, flow=args.flow, dist_type = args.dist_type, boundary_value = distance_boundary)
        k_boundary = determin_boundary(k_weight, share_height=share_height,macro_width=args.macro_width, flow=args.flow, dist_type = args.dist_type, boundary_value = distance_boundary)
        v_boundary = determin_boundary(v_weight, share_height=share_height,macro_width=args.macro_width, flow=args.flow, dist_type = args.dist_type, boundary_value = distance_boundary)
        q_bound.append(float(q_boundary))
        k_bound.append(float(k_boundary))
        v_bound.append(float(v_boundary))

        num_heads = 3072//macro_width ## fc1
        weight = model.blocks[i].mlp.fc1.weight
        fc1_boundary  = determin_boundary(weight,share_height=share_height,macro_width=args.macro_width, flow=args.flow, dist_type = args.dist_type, boundary_value = distance_boundary)
        fc1_bound.append(float(fc1_boundary))

        num_heads = 768//macro_width ## fc2
        weight = model.blocks[i].mlp.fc2.weight
        fc2_boundary = determin_boundary(weight, share_height=share_height_fc2, macro_width=args.macro_width,flow=args.flow, dist_type = args.dist_type, boundary_value = distance_boundary)
        fc2_bound.append(float(fc2_boundary))
    print("ratio of distance < ",distance_boundary)
    print("q distance : ",q_bound)
    print("k distance : ",k_bound)
    print("v distance : ",v_bound)
    print("fc1 distance : ",fc1_bound)
    print("fc2 distance : ",fc2_bound)
    
"""