import torch
from tqdm.autonotebook import tqdm

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from torchvision import transforms, utils
import copy
import os
import pickle

import json
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

def compute_column_distances_r1(A, B, dist_type = "euclidean"):
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
    else:
        raise ValueError("dist_type must be 'euclidean' or 'cosine'.")

    return distances

@torch.no_grad()
def row_sharing_r1(weight, distance_boundary, max_sharing_rate=0.5, return_shared_index = False, macro_height = 64, flow = "row", no_sharing = False, macro_width = 64, dist_type = "euclidean", share_height = 768, min_sharing_rate_per_macro = 0.7):
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
    
@torch.no_grad()
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
    


def weight_share_all(model,qkv_ratio,fc1_ratio,fc2_ratio, no_sharing=False,macro_width=64,args=None,distance_boundary = 100.0, set_mask = True):
    boundary_list = [distance_boundary]*12
    sharing_block_list = []
    train_block_list = []
    # sharing_rate_list : [(qkv_ratio*block_ratio)*4, (qkv_ratio)*4, (qkv_ratio*(2-block_ratio))*4]
    sharing_rate_list = [qkv_ratio*args.block_ratio]*4 + [qkv_ratio]*4 + [qkv_ratio*(2-args.block_ratio)]*4
    
    dim = 768
    #idx_mapping = []
    qkv_mask = None
    fc1_mask = None
    fc2_mask = None
    if args.share_height_type == "macro":
        share_height = args.macro_height
    elif args.share_height_type == "whole":
        share_height = 768
    else:
        raise Exception("share_height_type should be macro or whole")

    if qkv_ratio > 0:
        start_time = time.time()
        qkv_mask = []
        mask_diff_list = []
        for i in range(12):
            weight = model.blocks[i].attn.qkv.weight
            q_weight = weight[:dim,:]
            k_weight = weight[dim:dim*2,:]
            v_weight = weight[dim*2:,:]
            q_weight_share, num_q_sharing, q_mask, mask_diff, num_q_train = row_sharing_r1(q_weight, distance_boundary= boundary_list[i], max_sharing_rate=sharing_rate_list[i], 
                                                                     return_shared_index=True, macro_height=args.macro_height,flow=args.flow, no_sharing=no_sharing, 
                                                                     macro_width=args.macro_width,dist_type = args.dist_type, share_height = share_height, min_sharing_rate_per_macro = args.min_sharing_rate_per_macro)
            mask_diff_list.append(mask_diff)
            k_weight_share, num_k_sharing, k_mask, mask_diff, num_k_train = row_sharing_r1(k_weight, distance_boundary= boundary_list[i], max_sharing_rate=sharing_rate_list[i], 
                                                                     return_shared_index=True, macro_height=args.macro_height,flow=args.flow, no_sharing=no_sharing, 
                                                                     macro_width=args.macro_width,dist_type = args.dist_type, share_height = share_height, min_sharing_rate_per_macro = args.min_sharing_rate_per_macro)
            mask_diff_list.append(mask_diff)
            v_weight_share, num_v_sharing, v_mask, mask_diff, num_v_train = row_sharing_r1(v_weight, distance_boundary= boundary_list[i], max_sharing_rate=sharing_rate_list[i], 
                                                                     return_shared_index=True, macro_height=args.macro_height,flow=args.flow, no_sharing=no_sharing, 
                                                                     macro_width=args.macro_width,dist_type = args.dist_type, share_height = share_height, min_sharing_rate_per_macro = args.min_sharing_rate_per_macro)
            mask_diff_list.append(mask_diff)
            new_weight = torch.cat([q_weight_share,k_weight_share,v_weight_share], dim=0)
            if not no_sharing:
                model.blocks[i].attn.qkv.weight.data = new_weight
                # model.blocks[i].attn.qkv.weight.requires_grad = True
                # model.blocks[i].attn.qkv.weight.grad = torch.zeros_like(new_weight, device=new_weight.device, dtype=new_weight.dtype)
            #idx_mapping.append([q_idx_map, k_idx_map, v_idx_map])

            qkv_mask.append([q_mask, k_mask, v_mask])
            sharing_block_list.append([num_q_sharing, num_k_sharing, num_v_sharing])
            train_block_list.append([num_q_train, num_k_train, num_v_train])

        # print("q_id_map:",q_idx_map)
        # print("k_id_map:",k_idx_map)
        # print("v_id_map:",v_idx_map)
        print("use time : ",time.time()-start_time)
        print(f"sharing_block_list : {sharing_block_list}")
        print(f"train_block_list : {train_block_list}")
        print("mask diff average : ",sum(mask_diff_list)/len(mask_diff_list))
        # with open("qkv_mask.txt", "w") as f:
        #     for block_id in range(12):
        #         for qkv in range(3):
        #             for head_id in range(11):
        #                 f.write(str(qkv_mask[block_id][qkv][head_id].tolist())+"%i, %i, %i"%(block_id,qkv,head_id)+"\n")

        # with open("qkv_mask_pickle.bin", "wb") as f:
        #     pickle.dump(qkv_mask, f)

        #weight_sharing_qkv(model, idx_mapping)


    # fc1 weight
    boundary_list = [distance_boundary]*12
    sharing_rate_list = [fc1_ratio*args.block_ratio]*4 + [fc1_ratio]*4 + [fc1_ratio*(2-args.block_ratio)]*4
    sharing_block_list = []
    train_block_list = []
    fc1_mask = None

    if args.share_height_type == "macro":
        share_height = args.macro_height
    elif args.share_height_type == "whole":
        share_height = 768
    else:
        raise Exception("share_height_type should be macro or whole")

    if fc1_ratio > 0:
        fc1_mask = []
        sharing_block_list = []
        mask_diff_list = []
        start_time = time.time()
        for i in range(12):
            weight = model.blocks[i].mlp.fc1.weight
            weight, num_sharing, mask, mask_diff, num_train = row_sharing_r1(weight, distance_boundary= boundary_list[i], max_sharing_rate=sharing_rate_list[i], 
                                                                 return_shared_index=True, macro_height=args.macro_height,flow=args.flow, no_sharing=no_sharing,
                                                                 macro_width=args.macro_width,dist_type = args.dist_type, share_height = share_height, min_sharing_rate_per_macro = args.min_sharing_rate_per_macro)
            mask_diff_list.append(mask_diff)
            if not no_sharing:
                model.blocks[i].mlp.fc1.weight.data = weight
            sharing_block_list.append(num_sharing)
            train_block_list.append(num_train)
            fc1_mask.append(mask)
            #idx_mapping[i].append(fc1_idx)
        print("use time : ",time.time()-start_time)
        print(f"sharing_block_list : {sharing_block_list}")
        print(f"train_block_list : {train_block_list}")
        print("mask diff average : ",sum(mask_diff_list)/len(mask_diff_list))



    # fc2 weight
    boundary_list = [distance_boundary]*12
    sharing_rate_list = [fc2_ratio*args.block_ratio]*4 + [fc2_ratio]*4 + [fc2_ratio*(2-args.block_ratio)]*4
    sharing_block_list = []
    train_block_list = []

    if args.share_height_type == "macro":
        share_height = args.macro_height
    elif args.share_height_type == "whole":
        share_height = 3072
    else:
        raise Exception("share_height_type should be macro or whole")

    if fc2_ratio > 0:

        sharing_block_list = []
        fc2_mask = []
        mask_diff_list = []
        start_time = time.time()
        for i in range(12):
            weight = model.blocks[i].mlp.fc2.weight
            weight, num_sharing, mask, mask_diff, num_train = row_sharing_r1(weight, distance_boundary= boundary_list[i], max_sharing_rate=sharing_rate_list[i],
                                                                  return_shared_index=True, macro_height=args.macro_height,flow=args.flow, 
                                                                  no_sharing=no_sharing, macro_width=args.macro_width,dist_type = args.dist_type, share_height = share_height, min_sharing_rate_per_macro = args.min_sharing_rate_per_macro)
            if not no_sharing:
                model.blocks[i].mlp.fc2.weight.data = weight
            
            # upd_time_col = 3072//args.macro_width
            # share_time = num_heads * upd_time_col
            # total_share = args.macro_width * (share_time-1)
            sharing_block_list.append(num_sharing)
            fc2_mask.append(mask)
            mask_diff_list.append(mask_diff)
            train_block_list.append(num_train)

            #idx_mapping[i].append(fc2_idx)
        print("use time : ",time.time()-start_time)
        print(f"sharing_block_list : {sharing_block_list}")
        print(f"train_block_list : {train_block_list}")
        print("mask diff average : ",sum(mask_diff_list)/len(mask_diff_list))


    if set_mask:
        model.set_mask(qkv_mask, fc1_mask, fc2_mask, flow=args.flow, macro_width=args.macro_width, macro_height=args.macro_height, dist_type=args.dist_type, share_height_type=args.share_height_type)
    # save the mask as a pickle file
    # print("save mask as a pickle file")
    # # let mask on cpu
    # for i in range(12):
    #     for j in range(3):
    #         qkv_mask[i][j] = [mask.cpu() for mask in qkv_mask[i][j]]
    #     fc1_mask[i] = [mask.cpu() for mask in fc1_mask[i]]
    #     fc2_mask[i] = [mask.cpu() for mask in fc2_mask[i]]
    # with open("mask.pickle", "wb") as f:
    #     pickle.dump([qkv_mask, fc1_mask, fc2_mask], f)
    

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
    