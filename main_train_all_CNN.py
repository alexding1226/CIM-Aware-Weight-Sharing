import argparse
from CNN.CNN_utils import *
import CNN.CNN_losses as losses
from CNN.CNN_weight_grad_share import *

import torch
from CNN.CNN_Imagenet_dataset import CNN_ImagenetDataset

from models.VGG import vgg16
from models.VGG_teacher import vgg16_teacher

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, RandomSampler


def get_parser():
    parser = argparse.ArgumentParser(description='CIM row wise sharing for CNN')

    parser.add_argument('--device', type=str, default='cuda', metavar='D')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log_dir', default="./log", type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--load_checkpoint', type=str,default=None)

    # Model Config
    parser.add_argument('--model_type',default="VGG16", type=str)

    # Loss Function
    parser.add_argument('--pruning_weight', type=float, default=1, metavar='W')

    # Validation
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--val_before_share', action='store_true')
    parser.add_argument('--val_batch_size', type=int, default=512, metavar='B')
    parser.add_argument('--reduced_val', action='store_true')
    parser.add_argument('--log_image', action='store_true')



    # Training
    # Training augmentation
    parser.add_argument("--train_aug", action="store_true")
    
    parser.add_argument('--epoch', type=int, default=10, metavar='E')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=5e-6, metavar='lr')
    parser.add_argument('--train_subset_size', type=int, default=-1, metavar='N')
    parser.add_argument('--train_batch_size', type=int, default=64, metavar='B')
    parser.add_argument('--save_checkpoint', type=str, default='checkpoint.pt')
    # MARK: Seems no use
    """ 
    parser.add_argument('--train_classifier', action='store_true')   
    parser.add_argument('--train_selectors', action='store_true')
    """
    parser.add_argument('--best_acc', type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    # Use CE or dist
    parser.add_argument("--loss_type", type=str, default="dist")

    parser.add_argument("--lr_factor", type=float, default=0.7)
    parser.add_argument("--lr_patience", type=int, default=3)

    parser.add_argument("--check_distance_value", type=float, default=-1)

    parser.add_argument("--pred_weight", type=float, default=1.0)
    parser.add_argument("--soft_weight", type=float, default=0.0)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    

    parser.add_argument("--Ar", type=int, default=1)
    parser.add_argument("--no_share_initial", action="store_true")
    parser.add_argument("--start_epoch",type=int,default=0)

    parser.add_argument("--share_every", type=int, default=15)
    parser.add_argument("--start_share_epoch", type=int, default=15)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--macro_width", type=int, default=16)
    parser.add_argument("--macro_height", type=int, default=64)
    parser.add_argument("--share_height_type", type=str, default="whole") # whole or macro. whole means the sharing height meets the weight height, macro means the sharing height is the macro height
    parser.add_argument("--flow", type=str, default="row") # define the direction of sharing, row or column
    parser.add_argument("--boundary",type=float,default=100.0)
    parser.add_argument("--min_sharing_rate_per_macro",type=float,default=0.8) # ex: 0.8 means that at least 0.8 * share_ratio of rows in one macro should be shared (0.8 * 0.5 = 0.4, 26 in 64 rows should be shared as a minimum amount)

    parser.add_argument("--dist_type", type=str, default="euclidean")
    # parser.add_argument("--block_ratio", type=float, default=1.0) # first 4 blocks are block ratio * share_ratio, middle 4 blocks are share_ratio, last 4 blocks are (2 - block ratio) * share_ratio

    # parser.add_argument("--load_qkv_mask", type=str, default=None)

    parser.add_argument("--saving_every_step", type=int, default=2000)
    

    parser.add_argument("--conv_ratio", type=float, default=0.05)
    parser.add_argument("--fc_ratio", type=float, default=0.05)

    parser.add_argument("--conv_ratio_list", type=float, nargs='+', default=[])
    parser.add_argument("--fc_ratio_list", type=float, nargs='+', default=[])


    # # add ratio scheduler
    parser.add_argument("--max_conv_ratio", type=float, default=0.5)
    parser.add_argument("--max_fc_ratio", type=float, default=0.5)    

    parser.add_argument("--ratio_change_epoch", type=int, default=0) # 0 means no change
    parser.add_argument("--ratio_change_step", type=int, default=4000) # how many steps to change the ratio
    parser.add_argument("--max_ratio_epoch", type=int, default=5)

    return parser

def get_dataloader(catlog_path, subset_size=None, batch_size=16, augamentation=False):
    root_path = '/home/common/SharedDataset/ImageNet'
    num_workers = min(32, batch_size)
    dataset = CNN_ImagenetDataset(root_path, catlog_path, augamentation=augamentation)
    print("dataset length:",len(dataset))
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
    print("dataloader length:",len(dataloader))
    return dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_ratio_list(args_ratio_list = [], desired_length = 0, to_pad = 0):
    if len(args_ratio_list) == 0:
        return [to_pad] * desired_length
    else:
        if len(args_ratio_list) < desired_length:
            return args_ratio_list + [to_pad] * (desired_length - len(args_ratio_list))
        else:
            return args_ratio_list[:desired_length]


def main():

    set_seed(2357)
    parser = get_parser()
    args = parser.parse_args()
    device = torch.device(args.device)


    # File Path
    val_catlog = 'val_list_10k.txt' if args.reduced_val else 'val_list.txt'
    PATH = '/home/remote/LDAP/r13_pony-1000035/ckpt/vgg16-397923af.pth' if args.model_type == 'VGG16' else None
    # PATH = '/home/remote/LDAP/r13_pony-1000035/ckpt/vgg16_features-amdegroot-88682ab5.pth' if args.model_type == 'VGG16' else None
    checkpoint = torch.load(PATH, weights_only=True)    

    model = vgg16()           ; model.to(device)   ; model.eval()   ; model.load_state_dict(checkpoint)
    teacher = vgg16_teacher() ; teacher.to(device) ; teacher.eval() ; teacher.load_state_dict(checkpoint)
    
    loss_fn = losses.CNNLoss(
        pred_weight=args.pred_weight, 
        soft_weight=args.soft_weight, 
        dist_weight=args.dist_weight, 
        Ar=args.Ar,
        teacher=teacher
    )

    print("Conv2D:")
    total_conv_layers = 0
    total_fc_layers = 0
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            total_conv_layers += 1
            print("\t", layer.weight.shape)
    print("Linear:")
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            total_fc_layers += 1
            print("\t", layer.weight.shape)

    conv_ratio_list = get_ratio_list(args.conv_ratio_list, total_conv_layers, args.conv_ratio)
    fc_ratio_list = get_ratio_list(args.fc_ratio_list, total_fc_layers, args.fc_ratio)

    args.conv_ratio_list = conv_ratio_list
    args.fc_ratio_list = fc_ratio_list

    print(f"Conv ratio list: {args.conv_ratio_list}")
    print(f"FC ratio list: {args.fc_ratio_list}")


    last_progress = torch.load('progress.pt') if args.resume else {}
    start_epoch = last_progress['epoch'] if args.resume else args.start_epoch

    if args.resume:
        model.load_state_dict(last_progress['model'])
        model.to(device)
    elif args.load_checkpoint:
        print("load checkpoint : ", args.load_checkpoint)
        model.load_state_dict(torch.load(args.load_checkpoint))
        model.to(device)
    

    if args.val_before_share:
        val_dataloader = get_dataloader(val_catlog, batch_size=args.val_batch_size)
        print(validate(model, device, val_dataloader, loss_fn, start_epoch))
        return

    print("start sharing")
    weight_share_vgg(
        model=model,
        conv_ratio_list = conv_ratio_list,
        fc_ratio_list = fc_ratio_list,
        no_sharing=args.no_share_initial,
        macro_width=args.macro_width,
        args=args,distance_boundary=args.boundary
    )
    
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


        train_epochs(
            model, 
            args.device, 
            train_dataloader, 
            val_dataloader, 
            eval_dataloader, 
            loss_fn, 
            optimizer, 
            scheduler, 
            teacher=teacher, 
            epochs=(start_epoch+1, args.epoch), 
            current_best_acc=best_acc, 
            log_dir=args.log_dir, 
            checkpoint=args.save_checkpoint, 
            epoch_callback=epoch_callback,
            checkpoint_dir=args.checkpoint_dir,
            args=args
        )


    # for module in model.modules():
    #     print(module)

    # for name, W in model.named_parameters():
    #     print(name)
    """
    features.0.weight
    features.0.bias
    features.2.weight
    features.2.bias
    features.5.weight
    features.5.bias
    features.7.weight
    features.7.bias
    features.10.weight
    features.10.bias
    features.12.weight
    features.12.bias
    features.14.weight
    features.14.bias
    features.17.weight
    features.17.bias
    features.19.weight
    features.19.bias
    features.21.weight
    features.21.bias
    features.24.weight
    features.24.bias
    features.26.weight
    features.26.bias
    features.28.weight
    features.28.bias
    classifier.0.weight
    classifier.0.bias
    classifier.3.weight
    classifier.3.bias
    classifier.6.weight
    classifier.6.bias
    """


if __name__ == '__main__':
    main()