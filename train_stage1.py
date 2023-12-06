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
logger = logging.getLogger("train_stage1.py")

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
    rgb = preds["rgb"].detach().permute(0, 3, 1, 2)
    rgb_gt = batch["image"]
    rgb_boxes = preds["rgb_boxes"].detach().permute(0, 3, 1, 2)
    img = make_grid(th.cat([rgb, rgb_gt, rgb_boxes], dim=2) / 255.0).clip(0.0, 1.0)
    save_image(img, path)


def main(config):
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

    model = load_from_config(
        config.model,
        assets=static_assets,
    )

    if config.checkpoint_path:
        state_dict = th.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device)

    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    loss_fn = load_from_config(config.loss).to(device)
    optimizer = load_from_config(config.optimizer, params=model.parameters())

    model_ddp = DDP(
        model, device_ids=[device], find_unused_parameters=True
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
            batch = to_device(batch, device)

            if local_rank == 0 and batch is None:
                logger.info(f"batch {b} is None, skipping")
                continue

            if local_rank == 0 and iteration >= config.train.n_max_iters:
                logger.info(f"stopping after {config.train.n_max_iters}")
                break

            preds = model_ddp(**batch, train_iter=iteration)

            # rendering and raymarching
            rm_preds = rm(
                prim_rgba=preds["prim_rgba"],
                prim_pos=preds["prim_pos"],
                prim_scale=preds["prim_scale"],
                prim_rot=preds["prim_rot"],
                RT=batch["Rt"],
                K=batch["K"],
            )

            rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)

            preds.update(
                alpha=rgba[..., -1].contiguous(), rgb=rgba[..., :3].contiguous()
            )

            loss, loss_dict = loss_fn(batch, preds, iteration)
            _loss_dict = process_losses(loss_dict)

            if th.isnan(loss):
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.warning(f"some of the losses is NaN, skipping: {loss_str}")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_rank == 0 and iteration % config.train.log_every_n_steps == 0:
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.info(f"epoch={epoch}, iter={iteration}: {loss_str}")

            if (
                local_rank == 0
                # and iteration
                and iteration % config.train.summary_every_n_steps == 0
            ):
                logger.info(
                    f"saving summary to {config.output_dir} after {iteration} steps"
                )
                with th.no_grad():
                    preds["rgb_boxes"] = render_mvp_boxes(rm, batch, preds)
                save_image_summary("{}/train_{:06d}.png".format(config.output_dir, iteration), batch, preds)

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

        pass


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
