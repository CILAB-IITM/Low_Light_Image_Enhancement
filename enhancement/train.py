import argparse
import yaml
import os
import utils
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import datasets
import models
import torch.nn as nn
from tqdm import tqdm
from test import eval_psnr
import matplotlib.pyplot as plt
from kornia.enhance import equalize, normalize
from models.loss import WarpLoss
from torchvision.transforms import Resize
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


torch.cuda.empty_cache()  #! Added
def make_data_loader(dataset_dict, tag=""):
    # check dict is not None
    #! dataset_dict = config.get("train_dataset") or config.get("train_dataset")
    # config.get('train_dataset') = [dataset, wrapper, batch_size]
    # dataset - [name, args]
    # name - image-folder-basic - Not entirely sure what this does
    # args
    #   - root_path_inp - path to the input folder
    #   - root_path_out -
    #   - split: train - Not sure what this does
    #       - repeat: 30 - Prolly the number of epochs
    #       - patchify: False
    #       - patch_size: 512
    #   wrapper:
    #     name: preprocessing
    #     args:
    #       patch: true
    #   batch_size: 4

    """
    dataset_dict['dataset'] - The Configuration for the train/val Dataset. Like name and args - This contains the path to the data
    folder as well

    dataset_dict["wrapper"] - Does the necessory pre processing on the dataset

    Not completely sure what datasets.make does, but for now, maybe not necessary

    dataset_dict['dataset'], dataset_dict['wrapper'] - this is configured using a YAML file. Most likely here is where we
    need to the right dataset

    """
    if dataset_dict is None:
        return None
    # All the processing happens here nenaikuran
    dataset = datasets.make(dataset_dict["dataset"])
    # print(len(dataset), 'Length of dataset after probably passing through image_folder', 'From train.py line 57')
    # print(dataset[0][1].shape, 'Shape of first input data sample', 'From train.py line 58')
    # print('dataset passed through image_folder', 'from train.py line 59')
    # if tag == "val":
    # print("Atleast it is being sent to the preprocessing module")
    # print('Till here no error')
    dataset = datasets.make(dataset_dict["wrapper"], args={"dataset": dataset})
    # print(len(dataset), 'Length of dataset after probably passing through wrapper', 'From train.py line 64')
    # print(dataset[0]['inp'].shape, 'Shape of first input data sample', 'From train.py line 65')
    # print('dataset passed through wrapper', 'from train.py line 66')
    # # print()
    # for i in range(2, len(dataset)):
    #     temp = dataset[i]
    #     print(i, 'Done')
    #     print()
    # print('Something wrong with dataloader only', 'From train.py line 69')



    # If we want to continue the training
    if config.get("resume") is None:
        log("{} dataset: size={}".format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log(" {}: shape={}".format(k, tuple(v.shape)))
    
    loader = DataLoader(
        dataset,
        batch_size=dataset_dict.get("batch_size"),
        shuffle=(tag == "train"),
        num_workers=16,
        pin_memory=False,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get("train_dataset"), tag="train")
    print('train_loader completed', 'from train.py line 79')
    val_loader = make_data_loader(config.get("val_dataset"), tag="val")
    # print('val_loader completed', 'from train.py line 79')
    return train_loader, val_loader


def prepare_training():
    """
    the YAML file has something called "resume" - prolly some way of resuming the training
    """
    if config.get("resume") is not None:
        # print(config.get("resume"), "Hiiii")  #!
        sv_file = torch.load(config["resume"])
        model = models.make(sv_file["model"], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file["optimizer"], load_sd=True
        )
        epoch_start = sv_file["epoch"] + 1
        if epoch_start <= config["epoch_max"]:
            if config.get("multi_step_lr") is None:
                lr_scheduler = None
            else:
                lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])
            for _ in range(epoch_start - 1):
                lr_scheduler.step()

    else:
        """
        models is also a library from hugging face
        utils - another library
        Basically we are configuring the model, optimiser and other stuff from the YAML file
        """
        model = models.make(config["model"]).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config["optimizer"])
        epoch_start = 1
        if config.get("multi_step_lr") is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])

        log("model: #params={}".format(utils.compute_num_params(model, text=True)))

    return model, optimizer, epoch_start, lr_scheduler


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
                (equalize((inp[i, :3] * 0.5 + 0.5)) * 255)
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


def train(train_loader, model, optimizer, loss_fn, epoch):
    """
    tqdm is some library
    """
    model.train()


    train_loss = utils.Averager()
    #! Is this for normalizing the data?
    data_norm = config["data_norm"]
    t = data_norm["inp"]
    inp_sub = torch.FloatTensor(t["sub"]).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t["div"]).view(1, -1, 1, 1).cuda()
    t = data_norm["gt"]
    gt_sub = torch.FloatTensor(t["sub"]).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t["div"]).view(1, 1, -1).cuda()

    i = 0
    for batch in tqdm(train_loader, leave=False, desc="train"):
        # print('Successfully a batch is prepared', 'from train.py line 336')
        for k, v in batch.items():
            batch[k] = v.float().cuda()

        inp = (batch["inp"] - inp_sub) / inp_div
        gt = (batch["out"] - gt_sub) / gt_div
        # print(gt.shape, 'GroundTruth Shape')
        # print(inp.shape, "input shape")  #! Not needed Shoudl delete later

        """
        A lot of changes made here. Check with the  origianl
        """
        # print(inp.shape, 'Input Shape this is from train from line 369')
        # print(gt.shape, 'GroundTruth Shape from train.py line 370')
        # print(inp.shape, 'Input shape to the model')
        pred = model(inp)

        # print(inp.shape, gt.shape, pred.shape, 'From train.py line 374')
        # pred = pred.clamp_(0, 1) # This feels bit sus
        if config.get('stereo'):
            resizer = Resize([192, 624])
            gt = resizer(gt)
        if i%100 == 0:
            print()
            print('Saving')
            plt.imshow(torch.permute(pred[0].clamp_(0,1), (1, 2 , 0)).detach().cpu().numpy())
            path = '/home/gpu/girish/enhancement/dummy'
            file_name = '{}.png'.format(str(i))
            plt.imsave(os.path.join(path, file_name), torch.permute(pred[0], (1, 2 , 0)).detach().cpu().numpy())

        
        i += 1

        # loss calculation
        if config["model"]["name"] == "cfnet":
            loss = loss_fn(
                pred["recons"],
                pred["pred"],
                gt,  
                inp,  
                delta=config["delta"],
                weights=config.get("weights"),
            )
        else:
            # print(gt, "check here")
            resizer = Resize([512, 512])
            pred = resizer(pred)
            # print(pred.shape, gt.shape, 'Pred and gt shape from train.py line 402')
            #! Can try putting this in a Class statement
            # print('loss calculated', 'from train.py line 381')

            loss = loss_fn(pred, gt)
            # print(gt.shape)
            # pred_byte = (pred.clamp_(0,1)*255).clamp_(0,255).byte()
            # gt_byte = (gt.clamp_(0,1)*255).clamp_(0,255).byte()
            # msssim = MultiScaleStructuralSimilarityIndexMeasure().to(device='cuda')
            # loss += 0.1*msssim(pred, gt)
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # visualising predictions
        # if i % 10 == 0:
        #     if config["model"]["name"] == "cfnet":
        #         writer.add_figure(
        #             "Predictions",
        #             figure_fn(inp, gt, pred["pred"]),
        #             global_step=(epoch - 1) * len(train_loader) + i,
        #         )
        #     else:
        #         writer.add_figure(
        #             "Predictions",
        #             figure_fn(inp, gt, pred),
        #             global_step=(epoch - 1) * len(train_loader) + i,
        #         )
        # print(pred.shape, 'pred Shape')
        pred, loss = None, None

    return train_loss.item()


def main(config_, save_path, save_name):
    """
    what is epoch_val #!
    """
    global config, log, writer

    log, writer = utils.set_save_path(save_path)
    if os.path.exists(os.path.join(save_path, "config.yaml")):
        with open(os.path.join(save_path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        config = config_

    if config.get("loss") is None:
        loss_fn = nn.MSELoss()
    elif config["loss"] == "l1":
        loss_fn = nn.L1Loss()
    elif config["loss"] == "warping-l1":
        loss_fn = WarpLoss()
    # elif config["loss"] == 'ms-ssim-l1':
    #     loss_fn = nn.L1Loss() + MultiScaleStructuralSimilarityIndexMeasure()

    # checking if cuda is available
    if torch.cuda.is_available():
        print(
            "GPU Devices avaliable: {}-{}".format(
                torch.cuda.device_count(), torch.cuda.get_device_name()
            )
        )
        train_loader, val_loader = make_data_loaders()
        # print('Successfully dataloaders prepared', 'from train.py line 444')
        # print(len(train_loader), 'Length of train_loader', 'From train.py line 448')



        model, optimizer, epoch_start, lr_scheduler = prepare_training()
        
        n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if n_gpus > 1:
            model = nn.parallel.DataParallel(model)

        epoch_max = config["epoch_max"]
        epoch_val = config.get("epoch_val")
        epoch_save = config.get("epoch_save")
        if config.get("best_val") is None:
            max_val_v = -1e18
        else:
            max_val_v = config.get("best_val")

        timer = utils.Timer()

        for epoch in range(epoch_start, epoch_max + 1):
            t_epoch_start = timer.t()
            log_info = ["epoch {}/{}".format(epoch, epoch_max)]
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            train_loss = train(  #!
                train_loader, model, optimizer, loss_fn, epoch
            )  # training
            log_info.append("train: loss={:.4f}".format(train_loss))
            writer.add_scalar("Train loss", train_loss, epoch)
            if lr_scheduler is not None:  #!
                lr_scheduler.step()  #!

            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model

            model_spec = config["model"].copy()
            model_spec["sd"] = model_.state_dict()
            optimizer_spec = config["optimizer"].copy()
            optimizer_spec["sd"] = optimizer.state_dict()
            sv_file = {"model": model_spec, "optimizer": optimizer_spec, "epoch": epoch}

            torch.save(
                sv_file, os.path.join(save_path, "epoch_last.pth")
            )  # saving model

            config["resume"] = os.path.join(
                save_path, "epoch_last.pth"
            )  # used to resume training if stopped in between
            with open(os.path.join(save_path, "config.yaml"), "w") as f:
                yaml.dump(config, f, sort_keys=False)


            # Model getting saved in here
            if (epoch_save is not None) and (epoch % epoch_save == 0):
                torch.save(
                    sv_file, os.path.join(save_path, "epoch-{}.pth".format(epoch))
                )

            if (epoch_val is not None) and (epoch % epoch_val == 0 or epoch == 5):
                # validation score
                # print("Validation Routine")
                val_res, val_loss = eval_psnr(
                    val_loader,
                    model_,
                    writer,
                    epoch,
                    loss_fn,
                    config,
                    data_norm=config["data_norm"],
                )
                log_info.append("val: loss={:.4}".format(val_loss))
                writer.add_scalar("val_loss", val_loss, epoch)
                log_info.append("val: psnr={:.4}".format(val_res))
                writer.add_scalar("psnr", val_res, epoch)

                if val_res > max_val_v:
                    max_val_v = val_res
                    config["best_val"] = max_val_v
                    with open(os.path.join(save_path, "config.yaml"), "w") as f:
                        yaml.dump(config, f, sort_keys=False)
                    torch.save(
                        sv_file, os.path.join(save_path, "epoch-best-psnr.pth")
                    )  # saving best psnr model

            t = timer.t()
            prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
            t_epoch = utils.time_text(t - t_epoch_start)
            t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
            log_info.append("{} {}/{}".format(t_epoch, t_elapsed, t_all))

            log(", ".join(log_info))
            writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # pass the file path of configuration file in yaml format
    parser.add_argument("--config")

    # specify the gpus to be used
    parser.add_argument("--gpu", default="0")

    # pass the name of the model to be saved
    parser.add_argument("--name")

    # create a Namespace of the passed arguments
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    save_name = args.name

    # loading config into a dict
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if save_name is None:
        save_name = "_" + args.config.split("/")[-1][: -len(".yaml")]
    else:
        save_name = "_" + save_name

    # full path to save the model and log files
    save_path = os.path.join("./results_clrimg_resized", save_name)

    main(config, save_path, save_name)
