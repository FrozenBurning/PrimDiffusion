import os
import sys
import imageio

import torch as th
import numpy as np
from omegaconf import OmegaConf
import random
from dva.ray_marcher import RayMarcher, generate_colored_boxes
from primdiffusion.dataset.renderpeople_crossid_dataset import RenderPeopleSViewDataset
from dva.io import load_static_assets_crossid_smpl, load_from_config
from dva.utils import to_device
from dva.geom import make_postex, compute_tbn

import logging

device = th.device("cuda")

logger = logging.getLogger("visualize.py")


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

def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    print(f"Using random seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

def to_video_out(input):
    ndarr = input[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", th.uint8).numpy()
    return ndarr

def main(config):
    use_ddim = config.ddim
    device = th.device("cuda:0")
    th.cuda.set_device(device)

    static_assets = load_static_assets_crossid_smpl(config)

    inference_output_dir = f"{config.output_dir}/primdiffusion_interm_visualization"
    checkpoint_path = config.checkpoint_path
    os.makedirs(inference_output_dir, exist_ok=True)
    video_path = os.path.join(inference_output_dir, 'videos')
    os.makedirs(video_path, exist_ok=True)

    OmegaConf.save(config, os.path.join(inference_output_dir, "config.yml"))
    logger.info(f"saving results to {inference_output_dir}")

    logger.info(f"starting inference with the config: {OmegaConf.to_yaml(config)}")

    model = load_from_config(
        config.model,
        assets=static_assets,
    )

    print('loading checkpoint {}'.format(checkpoint_path))
    state_dict = th.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.to(device)
    model.device = device
    model.eval()

    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    dataset = RenderPeopleSViewDataset(
        **config.data,
        cameras=config.cameras_train,
        cond_cameras=config.cameras_cond,
        sample_cameras=False,
        is_train=False,
        camera_id='00',
    )

    sample_num = 1
    seed_list = [1007,]
    dataset.gen_inf_cameras(num_views=5)
    for iter in range(1000):
        logger.info('Rendering iteration-{:04d}......'.format(iter))
        set_random_seed(iter)
        batch = dataset.sample_cam_smpl()
        batch = to_device(batch, device)
        if use_ddim:
            log_every_t = 1
            samples, z_denoise_row = model.sample_log(cond=None, batch_size = sample_num, ddim=True, ddim_steps=100, eta=0.0, log_every_t=log_every_t)
            z_denoise_row = z_denoise_row['x_inter']
        else:
            log_every_t = 10
            samples, z_denoise_row = model.sample_log(cond=None, batch_size = sample_num, ddim=False, ddim_steps=None, eta=0.0, log_every_t=log_every_t)
        samples = (samples / model.scaling_factor + 1) / 2. * 255.
        denoise_row = (th.stack(z_denoise_row) / model.scaling_factor + 1) / 2. * 255
        prim_size = config.model.bodydecoder_config.prim_size
        n_prims_x = n_prims_y = int(config.model.bodydecoder_config.n_prims ** 0.5)

        # plot denoising row
        denoise_row = denoise_row.reshape(-1, sample_num, prim_size, 7, n_prims_y, prim_size, n_prims_x, prim_size).permute(0, 1, 4, 6, 3, 2, 5, 7).reshape(-1, sample_num, n_prims_y * n_prims_x, 7, prim_size, prim_size, prim_size)
        denoise_sample_deltascale = th.mean(denoise_row[:, :, :, 4:], dim=(-1, -2, -3)) / 255. * 20.
        denoise_sample_rgba = denoise_row[:, :, :, :4, :, :, :]
        
        num_steps = denoise_row.shape[0]
        for i in range(sample_num):
            batch = dataset.sample_cam_smpl()
            sam_cam = {}
            sam_cam.update(dataset.inf_cameras[dataset.subject_ids[0]]['camera0000'])
            for k, v in sam_cam.items():
                if isinstance(v, np.ndarray):
                    sam_cam[k] = v[None, ...]
            batch.update(sam_cam)
            batch = to_device(batch, device)
            B = 1
            geom = model.bodydecoder.lbs_fn(
                poses = batch["poses"],
                shapes = batch["shapes"],
                Rh = batch["Rh"],
                Th = batch["Th"],
                v_template = model.bodydecoder.lbs_fn.v_template[np.newaxis],
            ) * 1000.0
            prim_pos_mesh = (
                make_postex(geom, model.bodydecoder.prim_vidx_img, model.bodydecoder.prim_bary_img)
                .permute(0, 2, 3, 1)
                .reshape(-1, model.bodydecoder.n_prims, 3)
                .detach()
            )
            prim_scale_mesh = (
                model.bodydecoder.prim_scale[np.newaxis, :, np.newaxis].expand(B, -1, 3).detach().clone()
            )
            tbn = compute_tbn(geom, model.bodydecoder.geo_fn.vt, model.bodydecoder.prim_vidx_img, model.bodydecoder.prim_vtidx_img)
            prim_rot_mesh = (
                th.stack(tbn, dim=-2)
                .reshape(B, model.bodydecoder.n_prims, 3, 3)
                .permute(0, 1, 3, 2)
                .contiguous()
                .detach()
            )
            prim_scale_mesh[prim_scale_mesh < 350.] -= 70.

            if use_ddim:
                f_denoise_out = imageio.get_writer(os.path.join(video_path, 'seed{:05d}_denoise_ddim.mp4'.format(iter)), fps=30)
                f_view_out = imageio.get_writer(os.path.join(video_path, 'seed{:05d}_novelview_ddim.mp4'.format(iter)), fps=30)
            else:
                f_denoise_out = imageio.get_writer(os.path.join(video_path, 'seed{:05d}_denoise.mp4'.format(iter)), fps=30)
                f_view_out = imageio.get_writer(os.path.join(video_path, 'seed{:05d}_novelview.mp4'.format(iter)), fps=30)

            for j in range(num_steps):
                denoise_srgba = denoise_sample_rgba[j, i, ...][None, ...]
                denoise_sdelta = denoise_sample_deltascale[j, i, ...][None, ...]

                rm_preds = rm(
                    prim_rgba=denoise_srgba,
                    prim_pos=prim_pos_mesh,
                    prim_scale=prim_scale_mesh * denoise_sdelta,
                    prim_rot=prim_rot_mesh,
                    RT=batch["Rt"],
                    K=batch["K"],
                )
                denoise_render_rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
                preds = {
                    'prim_rot': prim_rot_mesh,
                    'prim_pos': prim_pos_mesh,        
                    'prim_scale': prim_scale_mesh * denoise_sdelta,
                    'prim_rgba': denoise_srgba,
                    }
                with th.no_grad():
                    denoise_mvp_box = render_mvp_boxes(rm, batch, preds)
                diffusion_rgb_path = os.path.join(inference_output_dir, 'seed{:05d}'.format(iter), 'diffusion_process')
                os.makedirs(diffusion_rgb_path, exist_ok=True)
                denoise_rgb_path = os.path.join(inference_output_dir, 'seed{:05d}'.format(iter), 'denoise_process')
                os.makedirs(denoise_rgb_path, exist_ok=True)
                denoise_rgb_image = (denoise_render_rgba[..., :3].contiguous().detach().permute(0, 3, 1, 2) / 255).clip(0.0, 1.0)
                denoise_box_image = (denoise_mvp_box.detach().permute(0, 3, 1, 2) / 255).clip(0.0, 1.0)
                
                denoise_rgb_image = to_video_out(denoise_rgb_image)
                denoise_box_image = to_video_out(denoise_box_image)
                f_denoise_out.append_data(np.concatenate((denoise_rgb_image, denoise_box_image), axis=1))
            
            for _ in range(30):
                f_denoise_out.append_data(np.concatenate((denoise_rgb_image, denoise_box_image), axis=1))
            
            f_denoise_out.close()
            # render novel view
            people_id = dataset.subject_ids[0]
            for camera_id in dataset.inf_cameras[people_id].keys():
                sam_cam = {}
                sam_cam.update(dataset.inf_cameras[people_id][camera_id])
                for k, v in sam_cam.items():
                    if isinstance(v, np.ndarray):
                        sam_cam[k] = v[None, ...]
                batch.update(sam_cam)
                batch = to_device(batch, device)
                denoise_srgba = denoise_sample_rgba[-1, 0, ...][None, ...]
                denoise_sdelta = denoise_sample_deltascale[-1, 0, ...][None, ...]
                rm_preds = rm(
                    prim_rgba=denoise_srgba,
                    prim_pos=prim_pos_mesh,
                    prim_scale=prim_scale_mesh * denoise_sdelta,
                    prim_rot=prim_rot_mesh,
                    RT=batch["Rt"],
                    K=batch["K"],
                )
                denoise_render_rgba = rm_preds["rgba_image"].permute(0, 2, 3, 1)
                preds = {
                    'prim_rot': prim_rot_mesh,
                    'prim_pos': prim_pos_mesh,        
                    'prim_scale': prim_scale_mesh * denoise_sdelta,
                    'prim_rgba': denoise_srgba,
                    }
                with th.no_grad():
                    denoise_mvp_box = render_mvp_boxes(rm, batch, preds)
                denoise_rgb_path = os.path.join(inference_output_dir, 'seed{:05d}'.format(iter), 'novelview')
                os.makedirs(denoise_rgb_path, exist_ok=True)
                denoise_rgb_image = (denoise_render_rgba[..., :3].contiguous().detach().permute(0, 3, 1, 2) / 255).clip(0.0, 1.0)
                denoise_box_image = (denoise_mvp_box.detach().permute(0, 3, 1, 2) / 255).clip(0.0, 1.0)
                denoise_rgb_image = to_video_out(denoise_rgb_image)
                denoise_box_image = to_video_out(denoise_box_image)
                f_view_out.append_data(np.concatenate((denoise_rgb_image, denoise_box_image), axis=1))

            f_view_out.close()

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
