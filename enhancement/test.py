import torch
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from kornia.enhance import equalize, normalize
import argparse
import os
import yaml
import datasets
import models
from torch.utils.data import DataLoader
import torch.nn as nn
from models.loss import WarpLoss
from torchvision.transforms import Resize


def figure_fn(inp, gt, pred, size=[256, 256]):
    if inp.shape[1] == 3:
        fig = plt.figure(figsize=(5 * 4, 5 * 4))
        for i in range(4):
            input_img = inp[i] * 0.5 + 0.5
            mean = torch.mean(input_img, dim=[1, 2])
            std = torch.std(input_img, dim=[1, 2])
            plt.subplot(4, 4, i * 4 + 1)
            plt.axis("off")
            plt.imshow(
                (equalize(input_img) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
            plt.subplot(4, 4, i * 4 + 2)
            plt.axis("off")
            plt.imshow(
                (normalize(input_img.unsqueeze(0), mean, std) * 255)
                .squeeze(0)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
            plt.subplot(4, 4, i * 4 + 3)
            plt.axis("off")
            plt.imshow(
                ((gt[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
            plt.subplot(4, 4, i * 4 + 4)
            plt.axis("off")
            plt.imshow(
                ((pred[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
        return fig
    elif inp.shape[1] == 6:
        fig = plt.figure(figsize=(5 * 3, 5 * 4))
        for i in range(3):
            plt.subplot(3, 4, i * 4 + 1)
            plt.axis("off")
            plt.imshow(
                (equalize(inp[i, :3] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
            plt.subplot(3, 4, i * 4 + 2)
            plt.axis("off")
            plt.imshow(
                ((inp[i, 3:] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
            plt.subplot(3, 4, i * 4 + 3)
            plt.axis("off")
            plt.imshow(
                ((gt[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
            plt.subplot(3, 4, i * 4 + 4)
            plt.axis("off")
            plt.imshow(
                ((pred[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy()
            )
        return fig
    elif inp.shape[1] == 1:
        fig = plt.figure(figsize=(5 * 4, 5 * 4))
        for i in range(4):
            input_img = inp[i] * 0.5 + 0.5
            mean = torch.mean(input_img, dim=[1, 2])
            std = torch.std(input_img, dim=[1, 2])
            plt.subplot(4, 4, i * 4 + 1)
            plt.axis("off")
            plt.imshow(
                (equalize(input_img) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
            plt.subplot(4, 4, i * 4 + 2)
            plt.axis("off")
            plt.imshow(
                (normalize(input_img.unsqueeze(0), mean, std) * 255)
                .squeeze(0)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
            plt.subplot(4, 4, i * 4 + 3)
            plt.axis("off")
            plt.imshow(
                ((gt[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
            plt.subplot(4, 4, i * 4 + 4)
            plt.axis("off")
            plt.imshow(
                ((pred[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
        return fig
    elif inp.shape[1] == 2:
        fig = plt.figure(figsize=(5 * 4, 5 * 4))
        for i in range(4):
            plt.subplot(4, 4, i * 4 + 1)
            plt.axis("off")
            plt.imshow(
                ((inp[i, :1] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
            plt.subplot(4, 4, i * 4 + 2)
            plt.axis("off")
            plt.imshow(
                ((inp[i, 1:] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
            plt.subplot(4, 4, i * 4 + 3)
            plt.axis("off")
            plt.imshow(
                ((gt[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
            plt.subplot(4, 4, i * 4 + 4)
            plt.axis("off")
            plt.imshow(
                ((pred[i] * 0.5 + 0.5) * 255)
                .permute(1, 2, 0)
                .long()
                .cpu()
                .detach()
                .numpy(),
                "gray",
            )
        return fig


# I assume this need not be understood, just hope that this code is right
def eval_psnr(loader, model, writer, epoch, loss_fn, config, data_norm=None):
    model.train()

    if data_norm is None:
        data_norm = {"inp": {"sub": [0], "div": [1]}, "gt": {"sub": [0], "div": [1]}}
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    metric_fn = utils.calc_psnr
    val_res = utils.Averager()
    val_loss = utils.Averager()

    pbar = tqdm(loader, leave=False, desc="val")  #! What is this w.r.t testing?
    i = 0
    for batch in pbar:
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.float().cuda()

            inp = (
                batch["inp"] - inp_sub
            ) / inp_div  # 4 X 3 X 512 X 512 (Depends on the Config set as well)
            gt = (batch["out"] - gt_sub) / gt_div
            # print(inp.shape, "Input to the validation Set")
            # This is where the prediction is happening. So everything before is the image processing
            pred = model(inp)
            if config["model"]["name"] == "cfnet":
                loss = loss_fn(
                    pred["recons"],
                    pred["pred"],
                    gt,
                    inp,
                    delta=config["delta"],
                    weights=[1, 1, 1],
                )
                pred = pred["pred"] * gt_div + gt_sub
                pred.clamp_(0, 1)
            else:
                # resizer = Resize([2160, 4096])
                # pred = resizer(pred)

                # resizer = Resize([256, 256])
                # gt = resizer(gt)
                resizer = Resize([384, 1248])
                gt = resizer(gt)
                loss = loss_fn(pred, gt)
                # print(loss, "From Validation")
                pred = pred * gt_div + gt_sub
                pred.clamp_(0, 1)

            val_loss.add(loss.item())

            # # visualising predictions
            # if i % 10 == 0:
            #     writer.add_figure(
            #         "Test Predictions",
            #         figure_fn(inp, gt, pred),
            #         global_step=(epoch - 1) * len(loader) + i,
            #     )

            i += 1
            res = metric_fn(pred, resizer(batch["out"]))
            val_res.add(res.item())
            # print(val_res.item())

            pred, loss = None, None

    return val_res.item(), val_loss.item()


# Understood this


def make_data_loader(dataset_dict, tag=""):
    # check dict is not None
    if dataset_dict is None:
        return None

    dataset = datasets.make(dataset_dict["dataset"])
    dataset = datasets.make(dataset_dict["wrapper"], args={"dataset": dataset})

    loader = DataLoader(
        dataset,
        batch_size=dataset_dict.get("batch_size"),
        shuffle=(tag == "train"),  # This will be false. Thus no shuffling happens here
        num_workers=8,
        pin_memory=True,
    )

    return loader


# def make_data_loaders():
#     pass


def prepare_testing(model_path):
    """
    the YAML file has something called "resume" - prolly some way of resuming the training
    """
    if config.get("resume") is not None:
        sv_file = torch.load(config["resume"])
        model = models.make(sv_file["model"], load_sd=True).cuda()

    else:
        """
        models is also a library from hugging face
        utils - another library
        Basically we are configuring the model, optimiser and other stuff from the YAML file
        """
        #! Weights from /home/gpu/girish/results_clrimg_resized/_unet_basic2_25_new/epoch-best-psnr.pth
        #! This needs to be passed to torch.load
        model_weights_path = os.path.join(model_path, "epoch-best-psnr.pth")
        sv_file = torch.load(model_weights_path)
        model = models.make(sv_file["model"], load_sd=True).cuda()

        log("model: #params={}".format(utils.compute_num_params(model, text=True)))

    return model


# def train(train_loader, model, optimizer, loss_fn, epoch):
#     """
#     tqdm is some library
#     """
#     model.train()

#     train_loss = utils.Averager()
#     #! Is this for normalizing the data?
#     data_norm = config["data_norm"]
#     t = data_norm["inp"]
#     inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
#     inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
#     t = data_norm["gt"]
#     gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
#     gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

#     i = 0
#     for batch in tqdm(train_loader, leave=False, desc="train"):
#         for k, v in batch.items():
#             batch[k] = v.float().cuda()

#         inp = (batch["inp"] - inp_sub) / inp_div
#         gt = (batch["out"] - gt_sub) / gt_div

#         pred = model(inp)

#         # loss calculation
#         if config["model"]["name"] == "cfnet":
#             loss = loss_fn(
#                 pred["recons"],
#                 pred["pred"],
#                 gt,
#                 inp,
#                 delta=config["delta"],
#                 weights=config.get("weights"),
#             )
#         else:
#             loss = loss_fn(pred, gt)
#         train_loss.add(loss.item())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # visualising predictions
#         if i % 10 == 0:
#             if config["model"]["name"] == "cfnet":
#                 writer.add_figure(
#                     "Predictions",
#                     figure_fn(inp, gt, pred["pred"]),
#                     global_step=(epoch - 1) * len(train_loader) + i,
#                 )
#             else:
#                 writer.add_figure(
#                     "Predictions",
#                     figure_fn(inp, gt, pred),
#                     global_step=(epoch - 1) * len(train_loader) + i,
#                 )
#         i += 1
#         pred, loss = None, None

#     return train_loss.item()


def main(config_, save_path, save_name, model_path):
    """
    what is epoch_val #!
    """
    global config, log, writer

    log, writer = utils.set_save_path(save_path)
    if os.path.exists(os.path.join(save_path, "test_config.yaml")):
        with open(os.path.join(save_path, "test_config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = config_

    if config.get("loss") is None:
        loss_fn = nn.MSELoss()
    elif config["loss"] == "l1":
        loss_fn = nn.L1Loss()
    elif config["loss"] == "warping-l1":
        loss_fn = WarpLoss()

    # checking if cuda is available
    if torch.cuda.is_available():
        print(
            "GPU Devices avaliable: {}-{}".format(
                torch.cuda.device_count(), torch.cuda.get_device_name()
            )
        )
        #! val_loader I do not think it is needed
        # train_loader, val_loader = make_data_loaders()
        # train_loader = make_data_loaders() #! make_data_loader is just enough
        test_loader = make_data_loader(config.get("test_dataset"), tag="train")

        model = prepare_testing(model_path)  #! Till here the code should work good
        #! Model Successuly loaded?

        n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if n_gpus > 1:
            model = nn.parallel.DataParallel(model)

        test_res, test_loss = eval_psnr(  #! This function now
            test_loader,
            model,
            writer,
            1,  #! What is this epoch for?
            loss_fn,
            config,
            data_norm=config["data_norm"],
        )
        print(test_res, test_loss)
        log_info = []
        log_info.append("test: loss={:.4}".format(test_loss))
        writer.add_scalar("test_loss", test_loss)
        log_info.append("test: psnr={:.4}".format(test_res))
        writer.add_scalar("psnr", test_res)
        # if lr_scheduler is not None:
        # lr_scheduler.step()

        # if n_gpus > 1:
        # model_ = model.module
        # else:
        # model_ = model

        # model_spec = config["model"].copy()
        # model_spec["sd"] = model_.state_dict()
        # optimizer_spec = config["optimizer"].copy()
        # optimizer_spec["sd"] = optimizer.state_dict()
        # sv_file = {"model": model_spec, "optimizer": optimizer_spec, "epoch": epoch}

        # torch.save(
        # sv_file, os.path.join(save_path, "epoch_last.pth")
        # )  # saving model

        # config["resume"] = os.path.join(
        # save_path, "epoch_last.pth"
        # )  # used to resume training if stopped in between
        # with open(os.path.join(save_path, "config.yaml"), "w") as f:
        # yaml.dump(config, f, sort_keys=False)

        # if (epoch_save is not None) and (epoch % epoch_save == 0):
        # torch.save(
        # sv_file, os.path.join(save_path, "epoch-{}.pth".format(epoch))
        # )

        # if (epoch_val is not None) and (epoch % epoch_val == 0):
        # validation score
        # val_res, val_loss = eval_psnr(
        # val_loader,
        # model_,
        # writer,
        # epoch,
        # loss_fn,
        # config,
        # data_norm=config["data_norm"],
        # )
        # log_info.append("val: loss={:.4}".format(val_loss))
        # writer.add_scalar("val_loss", val_loss, epoch)
        # log_info.append("val: psnr={:.4}".format(val_res))
        # writer.add_scalar("psnr", val_res, epoch)

        # if val_res > max_val_v:
        # max_val_v = val_res
        # config["best_val"] = max_val_v
        # with open(os.path.join(save_path, "config.yaml"), "w") as f:
        # yaml.dump(config, f, sort_keys=False)
        # torch.save(
        # sv_file, os.path.join(save_path, "epoch-best-psnr.pth")
        # )  # saving best psnr model

        # t = timer.t()
        # prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        # t_epoch = utils.time_text(t - t_epoch_start)
        # t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        # log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

        log(", ".join(log_info))
        writer.flush()


# this script can be run separately to assess individual image psnr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #! path to the model needs to be sent
    #! .pth file contains the weights
    parser.add_argument("--model")  #! This is how the path should be
    #! /home/gpu/girish/results_clrimg_resized/_unet_basic2_25_new not to the weights
    parser.add_argument("--metric")

    parser.add_argument("--name")

    args = parser.parse_args()

    # make dataloader of testdata
    model_path = args.model
    metric = args.metric  #! This is just psnr. Not path and all
    save_name = args.name

    test_config_path = os.path.join(model_path, "test_config.yaml")
    with open(test_config_path, "r") as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)

    #! train_config['val_dataset'] is the path to the test dataset
    test_loader = make_data_loader(test_config["test_dataset"], tag="test")
    # print(train_config["val_dataset"])
    # print(args.model.split("/")[-1])
    # Save Path to be defined
    if save_name is None:
        save_name = "test" + args.model.split("/")[-1]
        # save_name = "test" + args.model.split("/")[-1][: -len(".yaml")]

    else:
        save_name = "test" + save_name
    # Save Name to be defined
    save_path = os.path.join("./results_clrimg_resized", save_name)
    main(test_config, save_path, save_name, model_path)

    # make model
    # sv_file = torch.load(os.path.join(model_path, "epoch-best-" + metric + ".pth"))
    # model = models.make(sv_file["model"], load_sd=True).cuda()
    # n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    # if n_gpus > 1:
    #     model = torch.parallel.DataParallel(model)

    #! <From here need to start coding>
