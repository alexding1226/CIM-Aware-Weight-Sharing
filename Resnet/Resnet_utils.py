import torch
from torch.utils.tensorboard import SummaryWriter

import os, time
from tqdm.autonotebook import tqdm

from Resnet.Resnet_weight_grad_share import *




def ratio_scheduler(initial_ratio, end_ratio, total_steps, end_step):
    scheduler_list = []
    for i in range(total_steps):
        if i < end_step:
            ratio = initial_ratio + (end_ratio - initial_ratio) * i / end_step
        else:
            ratio = end_ratio
        scheduler_list.append(ratio)
    return scheduler_list


def update_ratio_list(
    current_conv_ratio_list,
    current_fc_ratio_list,
    args,
    conv_ratio_step,
    fc_ratio_step,
):
    def get_new_ratio(current_ratio, max_ratio, ratio_step):
        return (
            current_ratio + ratio_step
            if current_ratio + ratio_step < max_ratio
            else max_ratio
        )
    def get_new_ratio_list(current_ratio_list, max_ratio, ratio_step):
        return [get_new_ratio(current_ratio, max_ratio, ratio_step) for current_ratio in current_ratio_list]

    new_conv_ratio_list = get_new_ratio_list(current_conv_ratio_list, args.max_conv_ratio, conv_ratio_step)
    new_fc_ratio_list = get_new_ratio_list(current_fc_ratio_list, args.max_fc_ratio, fc_ratio_step)
    return new_conv_ratio_list, new_fc_ratio_list


def validate(model, device, dataloader, loss_fn, epoch):
    model.to(device)
    model.eval()
    loss_fn.eval()

    agg_loss = 0
    agg_loss_detail = {}
    for loss_term in loss_fn.loss_terms():
        agg_loss_detail[loss_term] = 0.0
    agg_acc = 0.0
    data_num = 0.0

    moving_loss = []
    moving_acc = []
    dist = 0

    desc = "Validation %d" % epoch

    with tqdm(
        dataloader, desc=desc, total=len(dataloader), dynamic_ncols=True, leave=False
    ) as pbar:
        with torch.no_grad():
            for idx, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                output = model(x)

                loss, loss_detail = loss_fn(x, output, y)

                agg_loss += loss.item()
                for loss_term in loss_fn.loss_terms():
                    agg_loss_detail[loss_term] += loss_detail[loss_term]

                if isinstance(output, tuple):
                    y_pred = torch.argmax(output[0], dim=1)
                else:
                    y_pred = torch.argmax(output, dim=1)
                agg_acc += torch.sum(y == y_pred).item()

                data_num += x.size(0)

                moving_loss.append(agg_loss / data_num)
                if len(moving_loss) > 10:
                    moving_loss.pop(0)
                moving_acc.append(agg_acc / data_num)
                if len(moving_acc) > 10:
                    moving_acc.pop(0)
                if "dist_loss" in loss_detail:
                    dist = loss_detail["dist_loss"]
                pbar.set_postfix(
                    Loss="%7.5f" % (sum(moving_loss) / len(moving_loss)),
                    Acc="%6.3f" % (sum(moving_acc) / len(moving_acc)),
                    Dist="%6.3f" % (dist),
                )

    agg_loss /= data_num
    # for loss_term in loss_fn.loss_terms():
    #     agg_loss_detail[loss_term] /= data_num
    agg_acc /= data_num

    return agg_acc, agg_loss, agg_loss_detail


def log_to_tensorboard(writer, epoch, prefix, info):
    prefix = prefix or ""
    for data in info:
        if data["type"] == "scalar":
            writer.add_scalar(prefix + "/" + data["key"], data["value"], epoch)
        elif data["type"] == "image":
            writer.add_image(prefix + "/" + data["key"], data["value"], epoch)
        else:
            raise NotImplementedError
    writer.flush()


def train_one_epoch(
    model,
    device,
    dataloader,
    loss_fn,
    optimizer,
    epoch,
    teacher=None,
    after_share=True,
    val_dataloader=None,
    checkpoint_dir=None,
    scheduler=None,
    best_shared_acc=0,
    start_share_epoch=0,
    conv_ratio_list=[],
    fc_ratio_list=[],
    conv_ratio_step=0,
    fc_ratio_step=0,
    ratio_change_step=4000,
    args=None,
):
    model.to(device)
    model.train()
    loss_fn.train()

    desc = "Epoch %d" % epoch
    agg_loss = 0
    agg_loss_detail = {}
    for loss_term in loss_fn.loss_terms():
        agg_loss_detail[loss_term] = 0.0
    agg_acc = 0
    data_num = 0

    moving_loss = []
    moving_acc = []
    moving_dist = [0]
    add_dist = not after_share

    current_conv_ratio_list = conv_ratio_list
    current_fc_ratio_list = fc_ratio_list

    # print("len(dataloader): ", len(dataloader))

    saving_every = 2000
    best_acc = best_shared_acc

    with tqdm(
        dataloader, desc=desc, total=len(dataloader), dynamic_ncols=True, leave=False
    ) as pbar:
        for idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss, loss_detail = loss_fn(x, output, y, add_dist=add_dist)
            norm_loss = loss / x.size(0)
            norm_loss.backward()

            add_dist = True

            start_time = time.time()
            use_time = time.time() - start_time
            
            optimizer.step()

            agg_loss += loss.item()
            for loss_term in loss_fn.loss_terms():
                agg_loss_detail[loss_term] += loss_detail[loss_term]

            if isinstance(output, tuple):
                y_pred = torch.argmax(output[0], dim=1)
            else:
                y_pred = torch.argmax(output, dim=1)
            agg_acc += torch.sum(y == y_pred).item()

            data_num += x.size(0)

            moving_loss.append(agg_loss / data_num)
            if len(moving_loss) > 10:
                moving_loss.pop(0)
            moving_acc.append(agg_acc / data_num)
            if len(moving_acc) > 10:
                moving_acc.pop(0)
            if "dist_loss" in loss_fn.loss_terms():
                dist = loss_detail["dist_loss"]
            else:
                dist = 0
            moving_dist.append(dist)
            if len(moving_dist) > 10:
                moving_dist.pop(0)
            pbar.set_postfix(
                Loss="%7.5f" % (sum(moving_loss) / len(moving_loss)),
                Acc="%6.3f" % (sum(moving_acc) / len(moving_acc)),
                LR="%10.10f" % optimizer.param_groups[0]["lr"],
                Dist="%6.5f" % (sum(moving_dist) / len(moving_dist)),
                # Share_time='%6.3f' % use_time
            )

            if idx % saving_every == 0 and idx > 0:
                val_acc, val_loss, val_loss_detail = validate(
                    model, device, val_dataloader, loss_fn, epoch
                )
                print("Validation: Loss: %7.5f - Acc: %6.3f" % (val_loss, val_acc))
                # torch.save(model.state_dict(), checkpoint_dir + "/checkpoint_%i_%i.pt"%(epoch, idx))
                with open(checkpoint_dir + "/acc_log.txt", "a") as f:
                    f.write(
                        str(idx)
                        + ","
                        + str(val_acc)
                        + ","
                        + str(optimizer.param_groups[0]["lr"])
                        + "\n"
                    )
                    f.close()
                if val_acc > best_acc and epoch > start_share_epoch:
                    best_acc = val_acc
                    torch.save(
                        model.state_dict(), checkpoint_dir + "/checkpoint_best.pt"
                    )
                scheduler.step(val_acc)
            if idx % ratio_change_step == 0 and idx > 0 and epoch > start_share_epoch:
                current_conv_ratio_list, current_fc_ratio_list = update_ratio_list(
                    current_conv_ratio_list,
                    current_fc_ratio_list,
                    args,
                    conv_ratio_step,
                    fc_ratio_step
                )
                print("Current conv ratio list: ", current_conv_ratio_list)
                print("Current fc ratio list: ", current_fc_ratio_list)

                print("start sharing")
                weight_share_resnet(
                    model=model,
                    conv_ratio_list=current_conv_ratio_list,
                    fc_ratio_list=current_fc_ratio_list,
                    macro_width=args.macro_width,
                    args=args,
                    distance_boundary=args.boundary,
                    set_mask=True,
                )
                add_dist = False

    agg_loss /= data_num
    for loss_term in loss_fn.loss_terms():
        agg_loss_detail[loss_term] /= data_num
    agg_acc /= data_num

    print("End Weight Sharing Resnet")
    return (
        agg_acc,
        agg_loss,
        agg_loss_detail,
        best_acc,
        current_conv_ratio_list,
        current_fc_ratio_list
    )


def evaluate(model, device, dataloader):
    model.to(device)
    model.eval()

    with torch.no_grad():
        x, y = next(iter(dataloader))
        x, y = x.to(device), y.to(device)
        _, masks, _ = model(x)

        b, c, _, _ = x.size()
        x = (
            x.reshape(b, c, 14, 16, 14, 16)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b, 14 * 14, 16, 16, c)
        )

        # Denormalize
        x = (x + 1) * 0.5

        stages = [
            torch.clamp(
                x
                + 0.7
                * (1 - mask).view(b, 14 * 14, 1, 1, 1).expand(b, 14 * 14, 16, 16, 3),
                min=0,
                max=1,
            )
            .reshape(b, 14, 14, 16, 16, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, 224, 224)
            for mask in masks
        ]
    return [
        x.reshape(b, 14, 14, 16, 16, c)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(b, c, 224, 224)
    ] + stages


def epoch_callback(epoch, acc, loss, best_acc, model, optimizer, scheduler):
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict(),
            "acc": best_acc,
        },
        "progress.pt",
    )

    # scheduler.step(acc)


def train_epochs(
    model,
    device,
    train_dataloader,
    val_dataloader,
    eval_dataloader,
    loss_fn,
    optimizer,
    scheduler,
    epochs,
    teacher=None,
    current_best_acc=0,
    log_dir=None,
    checkpoint_interval=None,
    checkpoint=None,
    epoch_callback=None,
    checkpoint_dir=None,
    args=None,
):

    log_dir = log_dir or "%s" % time.ctime().replace(" ", "_")
    log_dir = "runs/" + log_dir
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_interval = checkpoint_interval or 1
    checkpoint = checkpoint or "checkpoint.pt"
    checkpoint = checkpoint_dir + "/" + checkpoint
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    with open(checkpoint_dir + "/args.txt", "w") as f:
        for arg in vars(args):
            f.write(str(arg) + ": " + str(getattr(args, arg)) + "\n")
        f.close()
    if type(epochs) is not tuple:
        epochs = (1, epochs)
    after_share = not args.no_share_initial

    model.to(device)

    current_conv_ratio_list = args.conv_ratio_list
    current_fc_ratio_list = args.fc_ratio_list

    print("Start train with:")
    print("Current conv ratio list: ", current_conv_ratio_list)
    print("Current fc ratio list: ", current_fc_ratio_list)


    step_num = len(train_dataloader) * (epochs[1] - epochs[0] + 1)
    print("len(dataloader): ", len(train_dataloader))
    print("Total step number: ", step_num)

    ratio_change_time = args.max_ratio_epoch * (
        len(train_dataloader) // args.ratio_change_step
    )
    print("Ratio change time: ", ratio_change_time)
    print("Ratio change step: ", args.ratio_change_step)

    if ratio_change_time == 0:
        conv_ratio_step = 0
        fc_ratio_step = 0
    else:
        conv_ratio_step = (args.max_conv_ratio - args.conv_ratio) / ratio_change_time
        fc_ratio_step = (args.max_fc_ratio - args.fcf_ratio) / ratio_change_time

    print("Conv2D ratio step: ", conv_ratio_step)
    print("FC ratio step: ", fc_ratio_step)

    if current_best_acc < 0:
        current_best_acc, _, loss_detail = validate(model, device, val_dataloader, loss_fn, epochs[0]-1)
        if "dist_loss" in loss_detail:
            print("Current distance loss: %6.3f" % (loss_detail["dist_loss"]))
    

    print("Current best acc: %6.3f" % current_best_acc)
    best_shared_acc = -1

    total_begin_time = time.time()
    with open(checkpoint_dir + "/acc_log.txt", "w") as f:
        # f.write("Epoch,Acc\n")
        # f.write(str(epochs[0]-1)+","+str(current_best_acc)+"\n")
        f.write("Epoch,Acc,LR\n")
        f.write(
            str(epochs[0] - 1)
            + ","
            + str(current_best_acc)
            + ","
            + str(optimizer.param_groups[0]["lr"])
            + "\n"
        )
        f.close()
    for epoch in range(epochs[0], epochs[1] + 1):
        # Training
        begin_time = time.time()
        (
            train_acc,
            train_loss,
            train_loss_detail,
            best_shared_acc,
            current_conv_ratio_list,
            current_fc_ratio_list,
        ) = train_one_epoch(
            model,
            device,
            train_dataloader,
            loss_fn,
            optimizer,
            epoch,
            teacher=teacher,
            after_share=after_share,
            val_dataloader=val_dataloader,
            checkpoint_dir=checkpoint_dir,
            scheduler=scheduler,
            best_shared_acc=best_shared_acc,
            start_share_epoch=args.start_share_epoch,
            conv_ratio_list=current_conv_ratio_list,
            fc_ratio_list=current_fc_ratio_list,
            conv_ratio_step=conv_ratio_step,
            fc_ratio_step=fc_ratio_step,
            ratio_change_step=args.ratio_change_step,
            args=args,
        )

        # print("="*20)
        # test_idx = 0
        # for layer in model.features:
        #     if isinstance(layer, nn.Conv2d):
        #         if test_idx == 12:
        #             print(layer.weight.data[:10])
        #         test_idx += 1
        # print("="*20)

        after_share = False

        info = []
        info.append({"type": "scalar", "key": "Loss", "value": train_loss})
        info.append({"type": "scalar", "key": "Accuracy", "value": train_acc})
        for loss_term in loss_fn.loss_terms():
            info.append(
                {
                    "type": "scalar",
                    "key": loss_term,
                    "value": train_loss_detail[loss_term],
                }
            )
        log_to_tensorboard(writer, epoch, "Training", info)

        if epoch % checkpoint_interval == 0:
            # Validation
            val_acc, val_loss, val_loss_detail = validate(
                model, device, val_dataloader, loss_fn, epoch
            )

            info = []
            info.append({"type": "scalar", "key": "Loss", "value": val_loss})
            info.append({"type": "scalar", "key": "Accuracy", "value": val_acc})
            for loss_term in loss_fn.loss_terms():
                info.append(
                    {
                        "type": "scalar",
                        "key": loss_term,
                        "value": val_loss_detail[loss_term],
                    }
                )
            log_to_tensorboard(writer, epoch, "Validation", info)

            if val_acc > best_shared_acc and epoch > args.start_share_epoch:
                print(
                    "Checkpoint saved. Acc: %6.3f -> %6.3f" % (best_shared_acc, val_acc)
                )
                best_shared_acc = val_acc
                torch.save(model.state_dict(), checkpoint)

            # # Evaluation
            # info = []
            # imgs = evaluate(model, device, eval_dataloader)
            # imgs = [img.unsqueeze(1) for img in imgs]
            # imgs = torch.stack(imgs, dim=1)
            # imgs = imgs.view(-1, 3, 224, 224)
            # imgs = utils.make_grid(imgs, 4, padding=0)
            # info.append({'type': 'image', 'key': 'Image', 'value': imgs})
            # log_to_tensorboard(writer, epoch, 'Evaluation', info)

            ellapsed = int(time.time() - begin_time)
            minutes = ellapsed // 60
            seconds = ellapsed % 60
            ellapsed = int(time.time() - total_begin_time)
            total_minutes = ellapsed // 60
            total_seconds = ellapsed % 60
            print(
                "Epoch %d: Loss-train: %7.5f - Loss-val: %7.5f - Acc-train: %6.3f - Acc-val: %6.3f - LR: %10.10f - Epoch Ellapsed: %02d:%02d - Total Ellapsed: %02d:%02d"
                % (
                    epoch,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                    optimizer.param_groups[0]["lr"],
                    minutes,
                    seconds,
                    total_minutes,
                    total_seconds,
                )
            )

            with open(checkpoint_dir + "/acc_log.txt", "a") as f:
                f.write(
                    str(epoch)
                    + ","
                    + str(val_acc)
                    + ","
                    + str(optimizer.param_groups[0]["lr"])
                    + "\n"
                )
                f.close()
            print("Loss Detail: ", end="")
            for loss_term in loss_fn.loss_terms():
                print("%s: %7.5f - " % (loss_term, val_loss_detail[loss_term]), end="")
            print()

            scheduler.step(val_acc)
        # if epoch_callback is not None:
        #     epoch_callback(epoch, val_acc, val_loss, current_best_acc, model, optimizer, scheduler)

        if epoch % args.save_every == 0:
            print("Last Checkpoint saved.")
            torch.save(model.state_dict(), checkpoint[:-3] + "_%i.pt" % epoch)

        if ((epoch - args.start_share_epoch) % args.share_every == 0) and (
            epoch >= args.start_share_epoch
        ):
            # weight_grad_share.check_distance(
            #     model=model,
            #     macro_width=args.macro_width,
            #     args=args,
            #     distance_boundary=args.check_distance_value,
            # )
            print("start sharing")
            weight_share_resnet(
                model=model,
                conv_ratio_list=current_conv_ratio_list,
                fc_ratio_list=current_fc_ratio_list,
                macro_width=args.macro_width,
                args=args,
                distance_boundary=args.boundary,
                set_mask=True,
            )
            # weight_grad_share.check_distance(
            #     model=model,
            #     macro_width=args.macro_width,
            #     args=args,
            #     distance_boundary=args.check_distance_value,
            # )

            print("Shared Checkpoint saved.")
            torch.save(model.state_dict(), checkpoint[:-3] + "_%i_shared.pt" % epoch)
            print(validate(model, device, val_dataloader, loss_fn, epoch))
            after_share = True

        if args.ratio_change_epoch > 0:
            if epoch % args.ratio_change_epoch == 0:
                current_conv_ratio_list, current_fc_ratio_list = update_ratio_list(
                    current_conv_ratio_list,
                    current_fc_ratio_list,
                    args,
                    conv_ratio_step,
                    fc_ratio_step,
                )

    val_acc, val_loss, val_loss_detail = validate(
        model, device, val_dataloader, loss_fn, epoch
    )
    print("Final acc: %6.3f" % val_acc)
    with open(checkpoint_dir + "/acc.txt", "w") as f:
        f.write(str(val_acc))
        f.close()

