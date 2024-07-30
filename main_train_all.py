import timm
import torch
from Imagenet_dataset import ImagenetDataset
import argparse
from utils import *
from weight_grad_share import *

from models.RSDeit_wo_distance import RSVisionTransformer as RSVisionTransformer_wo_distance
from models.RSDeit_distance_r1 import RSVisionTransformer as RSVisionTransformer_distance_r1

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from torch.utils.data import DataLoader, RandomSampler
import losses
import numpy as np
import time

import pickle

def get_parser():
    parser = argparse.ArgumentParser(description='CIM row wise sharing')

    parser.add_argument('--device', type=str, default='cuda', metavar='D')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log_dir', default="./log", type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--load_checkpoint', type=str,default=None)

    # Model Config
    parser.add_argument('--model_type',default="deit3_base_patch16_224", type=str)
    #parser.add_argument('--model_type',default="deit_base_patch16_224", type=str)

    # Loss Function
    parser.add_argument('--pruning_weight', type=float, default=1, metavar='W')

    # Validation
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--val_before_share', action='store_true')
    parser.add_argument('--val_batch_size', type=int, default=128, metavar='B')
    parser.add_argument('--reduced_val', action='store_true')
    parser.add_argument('--log_image', action='store_true')
    # Training
    parser.add_argument("--train_aug", action="store_true")
    parser.add_argument('--epoch', type=int, default=1, metavar='E')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='lr')
    parser.add_argument('--train_subset_size', type=int, default=-1, metavar='N')
    parser.add_argument('--train_batch_size', type=int, default=64, metavar='B')
    parser.add_argument('--save_checkpoint', type=str, default='checkpoint.pt')
    parser.add_argument('--train_classifier', action='store_true')
    parser.add_argument('--train_selectors', action='store_true')
    parser.add_argument('--best_acc', type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    parser.add_argument("--loss_type", type=str, default="dist")

    parser.add_argument("--train_qkv", action="store_true")
    parser.add_argument("--lr_factor", type=float, default=0.7)
    parser.add_argument("--lr_patience", type=int, default=3)

    parser.add_argument("--check_distance_value", type=float, default=-1)

    parser.add_argument("--qkv_weight", type=float, default=1.0)
    parser.add_argument("--pred_weight", type=float, default=0.0)
    parser.add_argument("--soft_weight", type=float, default=1.0)
    parser.add_argument("--hidden_weight", type=float, default=1.0)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument("--Ar", type=int, default=1)
    parser.add_argument("--no_share_initial", action="store_true")
    parser.add_argument("--start_epoch",type=int,default=0)

    parser.add_argument("--qkv_ratio", type=float, default=0.5)
    parser.add_argument("--fc1_ratio", type=float, default=0.5)
    parser.add_argument("--fc2_ratio", type=float, default=0.0)
    parser.add_argument("--share_every", type=int, default=15)
    parser.add_argument("--start_share_epoch", type=int, default=15)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--macro_width", type=int, default=16)
    parser.add_argument("--macro_height", type=int, default=64)
    parser.add_argument("--share_height_type", type=str, default="whole") # whole or macro. whole means the sharing height meets the weight height, macro means the sharing height is the macro height
    parser.add_argument("--flow", type=str, default="row") # define the direction of sharing, row or column
    parser.add_argument("--boundary",type=float,default=100.0)
    parser.add_argument("--min_sharing_rate_per_macro",type=float,default=0.8)

    parser.add_argument("--dist_type", type=str, default="euclidean")

    parser.add_argument("--load_qkv_mask", type=str, default=None)

    return parser

def get_dataloader(catlog_path, subset_size=None, batch_size=16, augamentation=False):
    root_path = '/home/common/SharedDataset/ImageNet'
    num_workers = min(32, batch_size)
    dataset = ImagenetDataset(root_path, catlog_path, augamentation=augamentation)
    print(len(dataset))
    if subset_size is None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=RandomSampler(dataset, num_samples=subset_size)
        )
    return dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device(args.device)



    set_seed(2357)
    if args.model_type == "deit3_base_patch16_224":
        model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True, init_values=1e-6)
    else:
        model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = RSVisionTransformer_distance_r1(**model_args)
    teacher = RSVisionTransformer_wo_distance(**model_args)
    model.to(device)
    teacher.to(device)
    model.eval()
    teacher.eval()
    model.copy_weights(args.model_type)
    teacher.copy_weights(args.model_type)
    dim = 768
    print("qkv shape:",model.blocks[0].attn.qkv.weight.shape) # [2304, 768]
    print("fc1 shape:",model.blocks[0].mlp.fc1.weight.shape) # [3072, 768]
    print("fc2 shape:",model.blocks[0].mlp.fc2.weight.shape) # [768, 3072]

    #loss_fn = torch.nn.CrossEntropyLoss()
    if args.loss_type == "ce":
        loss_fn = losses.CEloss()
    elif args.loss_type == "dist":
        # if args.with_dist:
            loss_fn = losses.Distlossqkv_hidden_dist(qkv_weight = args.qkv_weight,pred_weight = args.pred_weight,
                                            soft_weight = args.soft_weight,hidden_weight = args.hidden_weight,dist_weight=args.dist_weight,
                                            teacher = teacher,Ar = args.Ar)
        # else:
        #     loss_fn = losses.Distlossqkv_hidden(qkv_weight = args.qkv_weight,pred_weight = args.pred_weight,
        #                                     soft_weight = args.soft_weight,hidden_weight = args.hidden_weight,
        #                                     teacher = teacher,Ar = args.Ar)
    # elif args.loss_type == "wodist":
    #     loss_fn = losses.Distlossqkv_hidden(qkv_weight = args.qkv_weight,pred_weight = args.pred_weight,
    #                                         soft_weight = args.soft_weight,hidden_weight = args.hidden_weight,
    #                                         teacher = teacher,Ar = args.Ar)
    else:
        raise NotImplementedError
    last_progress = torch.load('progress.pt') if args.resume else {}
    start_epoch = last_progress['epoch'] if args.resume else args.start_epoch
    # model = timm.create_model(args.model_type, pretrained=True)
    # model.to(device)
    # model.eval()

    ## hardware resource : 768 * 64 * 3
    ## for fc1 : compute 768 * (64*3)
    ## for fc2 : compute 3072 * (64*3/4) = 3072 * 48
    ## for qkv : compute 768 * 64




    
    ## qkv weight
    boundary_list = [100]*12
    sharing_rate_list = [args.qkv_ratio] * 12
    sharing_block_list = []
    idx_mapping = None

    if args.resume:
        model.load_state_dict(last_progress['model'])
        model.to(device)

    elif args.load_checkpoint:
        print("load checkpoint : ", args.load_checkpoint)
        model.load_state_dict(torch.load(args.load_checkpoint))
        model.to(device)

    if args.check_distance_value > 0:
        check_distance(model=model,macro_width=args.macro_width,args=args, distance_boundary=args.check_distance_value)

    if args.val_before_share:
        val_catlog = 'val_list_10k.txt' if args.reduced_val else 'val_list.txt'
        val_dataloader = get_dataloader(val_catlog, batch_size=args.val_batch_size)
        print(validate(model, device, val_dataloader, loss_fn, start_epoch))
        return

    print("start sharing")

    if args.no_share_initial and args.load_qkv_mask is not None:
        print("load qkv mask : ", args.load_qkv_mask)
        with open(args.load_qkv_mask, 'rb') as f:
            qkv_mask = pickle.load(f)
            f.close()
        model.set_mask(qkv_mask, fc1_mask=None, fc2_mask=None, flow=args.flow, macro_width=args.macro_width, macro_height=args.macro_height, dist_type=args.dist_type)
    else:
        weight_share_all(model=model,qkv_ratio=args.qkv_ratio,fc1_ratio=args.fc1_ratio,fc2_ratio=args.fc2_ratio,no_sharing=args.no_share_initial,macro_width=args.macro_width,args=args,distance_boundary=args.boundary)

    if args.check_distance_value > 0:
        check_distance(model=model ,macro_width=args.macro_width,args=args, distance_boundary=args.check_distance_value)


    
    

    

    

    val_catlog = 'val_list_10k.txt' if args.reduced_val else 'val_list.txt'
    if args.validate:
        # Validation Only
        val_dataloader = get_dataloader(val_catlog, batch_size=args.val_batch_size)
        print(validate(model, device, val_dataloader, loss_fn, start_epoch))

    else:
        if args.train_subset_size > 0:
            train_dataloader = get_dataloader('train_list.txt', subset_size=args.train_subset_size, batch_size=args.train_batch_size, augamentation=args.train_aug)
        else:
            train_dataloader = get_dataloader('train_list.txt', batch_size=args.train_batch_size, augamentation=args.train_aug)
        val_dataloader = get_dataloader(val_catlog, batch_size=args.val_batch_size)
        eval_dataloader = get_dataloader(val_catlog, batch_size=args.val_batch_size)

        if args.train_qkv:
            model.freeze_parameters()
            model.unfreeze_qkv()

        optimizer = AdamW(model.parameters(), lr=args.lr)

        scheduler = ReduceLROnPlateau(optimizer, min_lr=args.min_lr, factor=args.lr_factor, patience=args.lr_patience, mode='max')

        if args.resume:
            optimizer.load_state_dict(last_progress['optimizer'])
            scheduler.load_state_dict(last_progress['scheduler'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if k != 'step':
                    state[k] = v.to(device)

        if args.resume:
            best_acc = last_progress['best_acc']
        elif args.best_acc is not None:
            best_acc = args.best_acc
        else:
            best_acc = -1

        
        
        idx_mapping = None



        train_epochs(model, args.device, train_dataloader, val_dataloader, eval_dataloader, loss_fn, 
                     optimizer, scheduler, teacher=teacher, epochs=(start_epoch+1, args.epoch), current_best_acc=best_acc, 
                     log_dir=args.log_dir, checkpoint=args.save_checkpoint, epoch_callback=epoch_callback,
                     checkpoint_dir=args.checkpoint_dir,args=args)
        

if __name__ == '__main__':
    main()