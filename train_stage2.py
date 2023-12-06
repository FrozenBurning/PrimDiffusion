import os
import sys

import torch as th
import numpy as np
from omegaconf import OmegaConf

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.data import DataLoader

from dva.ray_marcher import RayMarcher, generate_colored_boxes
from primdiffusion.dataset.renderpeople_crossid_dataset import RenderPeopleSViewDataset
from dva.io import load_static_assets_crossid_smpl, load_from_config
from dva.losses import process_losses
from dva.utils import to_device

from torchvision.utils import make_grid, save_image

import logging
from tensorboardX import SummaryWriter

device = th.device("cuda")

logger = logging.getLogger("train_stage2.py")


def render_mvp_boxes(rm, batch, preds):
    with th.no_grad():
        boxes_rgba = generate_colored_boxes(
            preds["prim_rgba"],
            preds["prim_rot"],
        )
        preds_boxes = rm(
            prim_rgba=boxes_rgba,
            prim_pos=preds["prim_pos"],
            prim_scale=preds["prim_scale"],
            prim_rot=preds["prim_rot"],
            RT=batch["Rt"],
            K=batch["K"],
        )

    return preds_boxes["rgba_image"][:, :3].permute(0, 2, 3, 1)


def save_image_summary(path, batch, preds):
    # TODO: process_fn here?
    if 'diffusion_rgb' in preds.keys():
        diffusion_rgb = preds["diffusion_rgb"].detach().permute(0, 3, 1, 2)
        rgb = preds["rgb"].detach().permute(0, 3, 1, 2)
        rgb_gt = batch["image"]
        rgb_boxes = preds["rgb_boxes"].detach().permute(0, 3, 1, 2)
        img = make_grid(th.cat([diffusion_rgb, rgb, rgb_gt, rgb_boxes], dim=2) / 255.0).clip(0.0, 1.0)
    else:
        rgb = preds["rgb"].detach().permute(0, 3, 1, 2)
        rgb_gt = batch["image"]
        rgb_boxes = preds["rgb_boxes"].detach().permute(0, 3, 1, 2)
        img = make_grid(th.cat([rgb, rgb_gt, rgb_boxes], dim=2) / 255.0).clip(0.0, 1.0)
    save_image(img, path)
    return img


def main(config):
    amp = config.train.amp
    scaler = th.cuda.amp.GradScaler() if amp else None

    dist.init_process_group("nccl")

    logging.basicConfig(level=logging.INFO)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = th.device(f"cuda:{local_rank}")
    th.cuda.set_device(device)

    static_assets = load_static_assets_crossid_smpl(config)
    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    OmegaConf.save(config, f"{config.output_dir}/config.yml")
    logger.info(f"saving results to {config.output_dir}")

    logger.info(f"starting training with the config: {OmegaConf.to_yaml(config)}")

    if local_rank == 0:
        writer = SummaryWriter(logdir=config.output_dir)
    model = load_from_config(
        config.model,
        assets = static_assets,
    )

    if config.pretrained_encoder:
        state_dict = th.load(config.pretrained_encoder, map_location='cpu')
        model.bodydecoder.load_state_dict(state_dict['model_state_dict'])
        logger.info(f"Loaded Pretrained encoder {config.pretrained_encoder}")

    if config.checkpoint_path:
        state_dict = th.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
        logger.info(f"Loaded checkpoint from {config.checkpoint_path}")
    
    model.device = device
    model = model.to(device)

    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    optimizer = load_from_config(config.optimizer, params=model.diffusion.parameters())

    model_ddp = th.nn.parallel.DistributedDataParallel(
        model, device_ids=[device]
    )

    dataset = RenderPeopleSViewDataset(
        **config.data,
        cameras=config.cameras_train,
        cond_cameras=config.cameras_cond,
    )

    train_sampler = th.utils.data.distributed.DistributedSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=config.train.get("batch_size", 4),
        pin_memory=False,
        sampler=train_sampler,
        num_workers=config.train.get("n_workers", 8),
        drop_last=True,
        worker_init_fn=lambda _: np.random.seed(),
    )

    iteration = 0

    for epoch in range(config.train.n_epochs):
        for b, batch in enumerate(loader):
            with th.cuda.amp.autocast(enabled=amp):
                batch = to_device(batch, device)
                if local_rank == 0 and batch is None:
                    logger.info(f"batch {b} is None, skipping")
                    continue
                if local_rank == 0 and iteration >= config.train.n_max_iters:
                    logger.info(f"stopping after {config.train.n_max_iters}")
                    break
                loss, loss_dict, preds = model_ddp(**batch, train_iter=iteration)
                _loss_dict = process_losses(loss_dict)
                if th.isnan(loss):
                    loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                    logger.warning(f"some of the losses is NaN, skipping: {loss_str}")
                    continue
                optimizer.zero_grad()
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            if local_rank == 0 and iteration % config.train.log_every_n_steps == 0:
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.info(f"epoch={epoch}, iter={iteration}: {loss_str}")
                for k, v in _loss_dict.items():
                    writer.add_scalar(k, v, iteration)
            if (
                local_rank == 0
                # and iteration
                and iteration % config.train.summary_every_n_steps == 0
            ):
                logger.info(
                    f"saving summary to {config.output_dir} after {iteration} steps"
                )
                sample_log = model.log_images(preds, N=2, n_row=2, plot_denoise_rows=True, ddim_steps=None, plot_diffusion_rows=True)
                diffusion_row = sample_log['diffusion_row']
                denoise_row = sample_log['denoise_row']
                step_num = denoise_row.shape[0]
                sample_num = denoise_row.shape[1]
                prim_size = config.model.bodydecoder_config.prim_size
                n_prims_x = n_prims_y = int(config.model.bodydecoder_config.n_prims ** 0.5)

                denoise_row = denoise_row.reshape(step_num, sample_num, prim_size, 7, n_prims_y, prim_size, n_prims_x, prim_size).permute(0, 1, 4, 6, 3, 2, 5, 7).reshape(step_num, sample_num, n_prims_y * n_prims_x, 7, prim_size, prim_size, prim_size)
                diffusion_row = diffusion_row.reshape(step_num, sample_num, prim_size, 7, n_prims_y, prim_size, n_prims_x, prim_size).permute(0, 1, 4, 6, 3, 2, 5, 7).reshape(step_num, sample_num, n_prims_y * n_prims_x, 7, prim_size, prim_size, prim_size)
                denoise_deltascale = denoise_row[:, :, :, 4:, 0, 0, 0] / 255. * 20.
                diffusion_deltascale = diffusion_row[:, :, :, 4:, 0, 0, 0] / 255. * 20.
                denoise_row = denoise_row[:, :, :, :4, :, :, :]
                diffusion_row = diffusion_row[:, :, :, :4, :, :, :]

                # rendering and raymarching
                preds["prim_pos"] = preds["prim_pos"][:sample_num, ...].repeat(step_num, 1, 1, 1)
                mesh_scale = preds["prim_mesh_scale"][:sample_num, ...].repeat(step_num, 1, 1, 1)
                preds["prim_rot"] = preds["prim_rot"][:sample_num, ...].repeat(step_num, 1, 1, 1, 1)
                batch["Rt"] = batch["Rt"][:sample_num, ...].repeat(step_num, 1, 1, 1)
                batch["K"] = batch["K"][:sample_num, ...].repeat(step_num, 1, 1, 1)

                with th.no_grad():
                    for sample_id in range(sample_num):
                        sample_batch = {}
                        sample_pred = {}
                        sample_batch["Rt"] = batch["Rt"][:, sample_id, ...].contiguous()
                        sample_batch["K"] = batch["K"][:, sample_id, ...].contiguous()
                        sample_pred["prim_pos"] = preds["prim_pos"][:, sample_id, ...].contiguous()
                        sample_pred["prim_scale"] = mesh_scale[:, sample_id, ...].contiguous() * denoise_deltascale[:, sample_id, ...]
                        sample_pred["prim_rot"] = preds["prim_rot"][:, sample_id, ...].contiguous()


                        sample_rgba = denoise_row[:, sample_id, ...].contiguous()
                        rm_sample_preds = rm(
                            prim_rgba=sample_rgba,
                            prim_pos=sample_pred["prim_pos"],
                            prim_scale=sample_pred["prim_scale"],
                            prim_rot=sample_pred["prim_rot"],
                            RT=sample_batch["Rt"],
                            K=sample_batch["K"],
                        )
                        sample_pred["prim_rgba"] = sample_rgba
                        rgba = rm_sample_preds["rgba_image"].permute(0, 2, 3, 1)
                        sample_pred.update(alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous())
                        sample_pred["rgb_boxes"] = render_mvp_boxes(rm, sample_batch, sample_pred)
                        sample_batch['image'] = batch['image'][sample_id, ...].repeat(step_num, 1, 1, 1)

                        diffusion_sample_rgba = diffusion_row[:, sample_id, ...].contiguous()
                        sample_pred["prim_scale"] = mesh_scale[:, sample_id, ...].contiguous() * diffusion_deltascale[:, sample_id, ...]
                        # get the original decoded result
                        diffusion_sample_rgba[0, ...] = preds["prim_rgba"][sample_id, ...]
                        diffusion_rm_sample = rm(
                            prim_rgba=diffusion_sample_rgba,
                            prim_pos=sample_pred["prim_pos"],
                            prim_scale=sample_pred["prim_scale"],
                            prim_rot=sample_pred["prim_rot"],
                            RT=sample_batch["Rt"],
                            K=sample_batch["K"],
                        )
                        diffusion_rgba = diffusion_rm_sample["rgba_image"].permute(0, 2, 3, 1)
                        sample_pred.update(diffusion_alpha=diffusion_rgba[..., -1].contiguous(), diffusion_rgb=diffusion_rgba[..., :3].contiguous())
                        saved_img = save_image_summary("{}/train_{:06d}_{:02d}.png".format(config.output_dir, iteration, sample_id), sample_batch, sample_pred)
                        writer.add_image('samples_{:02d}'.format(sample_id), saved_img, iteration)

            if (
                local_rank == 0
                and iteration
                and iteration % config.train.ckpt_every_n_steps == 0
            ):
                logger.info(f"saving checkpoint after {iteration} steps")

                params = {
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                th.save(params, f"{config.output_dir}/checkpoints/{iteration:06d}.pt")

            iteration += 1

if __name__ == "__main__":
    th.backends.cudnn.benchmark = True
    # set config
    config = OmegaConf.load(str(sys.argv[1]))
    config_cli = OmegaConf.from_cli(args_list=sys.argv[2:])
    if config_cli:
        logger.info("overriding with following values from args:")
        logger.info(OmegaConf.to_yaml(config_cli))
        config = OmegaConf.merge(config, config_cli)

    main(config)
