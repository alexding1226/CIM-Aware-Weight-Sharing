import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# Load the data
# set device = cpu


with open('mask.pickle', 'rb') as f:
    # set device = cpu
    mask = pickle.load(f)
    qkv_mask = mask[0]
    fc1_mask = mask[1]
    fc2_mask = mask[2]

#convert list of list of tensor to tensor
q_masks = [qkv_mask[i][0] for i in range(len(qkv_mask))]
q_masks = torch.stack([torch.stack(mask) for mask in q_masks], dim=0)
k_masks = [qkv_mask[i][1] for i in range(len(qkv_mask))]
k_masks = torch.stack([torch.stack(mask) for mask in k_masks], dim=0)
v_masks = [qkv_mask[i][2] for i in range(len(qkv_mask))]
v_masks = torch.stack([torch.stack(mask) for mask in v_masks], dim=0)
fc1_masks = [fc1_mask[i] for i in range(len(fc1_mask))]
fc1_masks = torch.stack([torch.stack(mask) for mask in fc1_masks], dim=0)
fc2_masks = [fc2_mask[i] for i in range(len(fc2_mask))]
fc2_masks = torch.stack([torch.stack(mask) for mask in fc2_masks], dim=0)

print("original shape")
print(q_masks.shape)
print(k_masks.shape)
print(v_masks.shape)
print(fc1_masks.shape)
print(fc2_masks.shape)

# Analyze the distribution of the mask
# divide 768 dimension into 12 groups
block_num, share_time, dim = q_masks.shape

q_masks = q_masks.view(block_num, -1 ,12, 64)
k_masks = k_masks.view(block_num, -1 ,12, 64)
v_masks = v_masks.view(block_num, -1 ,12, 64)
block_num, share_time, dim = fc1_masks.shape
fc1_masks = fc1_masks.view(block_num, -1 ,12, 64)
block_num, share_time, dim = fc2_masks.shape
fc2_masks = fc2_masks.view(block_num, -1 ,12, 64)

print("grouped shape")
print(q_masks.shape)
print(k_masks.shape)
print(v_masks.shape)
print(fc1_masks.shape)
print(fc2_masks.shape)

q_masks_count = torch.sum(q_masks, dim=3, dtype=torch.float32)
k_masks_count = torch.sum(k_masks, dim=3, dtype=torch.float32)
v_masks_count = torch.sum(v_masks, dim=3, dtype=torch.float32)
fc1_masks_count = torch.sum(fc1_masks, dim=3, dtype=torch.float32)
fc2_masks_count = torch.sum(fc2_masks, dim=3, dtype=torch.float32)
print("count shape")
print(q_masks_count.shape)
print(fc1_masks_count.shape)
print(fc2_masks_count.shape)

all_masks = torch.cat((q_masks_count, k_masks_count, v_masks_count, fc1_masks_count, fc2_masks_count), dim=1)
print("all_masks shape")
print(all_masks.shape)

mean = torch.mean(all_masks).item()
std = torch.std(all_masks).item()
max = torch.max(all_masks).item()
min = torch.min(all_masks).item()
print("mean: ", mean)
print("std: ", std)
print("max: ", max)
print("min: ", min)

plt.hist(all_masks.flatten().cpu().numpy(), bins=int((max-min)), range=(min, max),edgecolor='black')
#plt.show()
plt.savefig('mask_distribution.png')
plt.close()

all_masks = all_masks.view(-1,12)
print(all_masks.shape)
min_per_block = torch.min(all_masks, dim=1).values
print(min_per_block.shape)
mean = torch.mean(min_per_block).item()
print("mean of min per update: ", mean)
plt.hist(min_per_block.cpu().numpy(),  edgecolor='black', bins=int(torch.max(min_per_block).item()-torch.min(min_per_block).item()))
#plt.show()
plt.savefig('min_per_update.png')

# mean = torch.mean(fc2_masks_count).item()
# std = torch.std(fc2_masks_count).item()
# max = torch.max(fc2_masks_count).item()
# min = torch.min(fc2_masks_count).item()
# print("mean: ", mean)
# print("std: ", std)
# print("max: ", max)
# print("min: ", min)
# plt.hist(fc2_masks_count.flatten().cpu().numpy(), bins=int((max-min)), range=(min, max),edgecolor='black')
# plt.show()

# def print_distribution(mask_count, name):
#     print(name)
#     print("mean: ", torch.mean(mask_count).item())
#     print("std: ", torch.std(mask_count).item())
#     print("max: ", torch.max(mask_count).item())
#     print("min: ", torch.min(mask_count).item())
#     print("")

# block_num, share_time, dim = q_masks_count.shape
# for i in range(block_num):
#     print_distribution(q_masks_count[i], "q_masks_count")

    

