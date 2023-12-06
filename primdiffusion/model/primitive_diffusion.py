import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from functools import partial
from tqdm import tqdm

from primdiffusion.model.attcond_smpl_model import BodyDecoder

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class PrimDiffusion(nn.Module):
    def __init__(
        self,
        unet_config,
        bodydecoder_config,
        assets,
        smpl_gender='neutral',
        parameterization='x0',
        loss_type = 'l2',
        l_simple_weight = 1.,
        original_elbo_weight = 0.,
        v_posterior = 0.,
    ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.loss_type = loss_type
        self.l_simple_weight = l_simple_weight
        self.original_elbo_weight = original_elbo_weight
        self.v_posterior = v_posterior
        self.clip_denoised = False
        self.log_every_t = 200
        self.dims = unet_config.dims
        self.scaling_factor = unet_config.scaling_factor
        self.normalize_bystd = unet_config.normalize_bystd
        self.d_cond = unet_config.d_cond
        if self.dims == 2:
            assert unet_config.in_channels == (4 + 3)* bodydecoder_config.prim_size
        elif self.dims == 3:
            assert unet_config.in_channels == 4
        # 3D conv: 32*32*8^3*4 -> 256*256*8*4
        # 2D conv: 32*32*8^3*4 -> 256*256*32
        self.channels = int(unet_config.in_channels)
        self.image_size = int(bodydecoder_config.prim_size * (bodydecoder_config.n_prims ** 0.5))
        self.device = None

        self.bodydecoder = BodyDecoder(assets=assets, **bodydecoder_config)
        self.bodydecoder.eval()
        for p in self.bodydecoder.parameters():
            p.requires_grad = False
        self.diffusion = UNetModel(**unet_config)
        self.register_schedule()
    
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def forward(
        self,
        # SMPL / LBS stuff
        poses,
        shapes,
        Rh,
        Th,
        #
        cam_pos,
        # conditioning images and R,T
        cond_Rt,
        cond_K,
        cond_image,
        train_iter=0,
        **kwargs,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                prim_dict = self.bodydecoder(poses, shapes, Rh, Th, cam_pos, cond_Rt, cond_K, cond_image, train_iter, **kwargs)
        x = prim_dict['prim_rgba'].reshape(-1, self.bodydecoder.n_prims_y, self.bodydecoder.n_prims_x, 4, self.bodydecoder.prim_size, self.bodydecoder.prim_size, self.bodydecoder.prim_size) # supposed to be Bx64x64x16^3x4
        prim_scale = prim_dict['delta_scale'].reshape(-1, self.bodydecoder.n_prims_y, self.bodydecoder.n_prims_x, 3)
        prim_scale = prim_scale[..., None, None, None].repeat(1, 1, 1, 1, self.bodydecoder.prim_size, self.bodydecoder.prim_size, self.bodydecoder.prim_size) / 20. * 255.
        x = torch.cat([x, prim_scale], dim=3) #supposed to be Bx32x32x7x8x8x8
        if self.dims == 2:
            x = x.permute(0, 4, 3, 1, 5, 2, 6).contiguous().reshape(-1, self.bodydecoder.prim_size*(4+3), self.bodydecoder.n_prims_y * self.bodydecoder.prim_size, self.bodydecoder.n_prims_x * self.bodydecoder.prim_size)
        else:
            raise NotImplementedError('dims {} not supported'.format(self.dims))
        x =  (x / 255) * 2 - 1
        x = x * self.scaling_factor
        
        # diffusion step
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        c = None if self.d_cond is None else rearrange(prim_dict['emb_conv'], 'b c h w -> b (h w) c').contiguous()
        ddpm_loss, ddpm_loss_dict = self.p_losses(x, c, t)
        return ddpm_loss, ddpm_loss_dict, prim_dict
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()
        
        loss = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        loss_simple = self.l_simple_weight * loss.mean()
        loss_dict.update({'loss_simple': loss.mean()})

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({'loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb
        loss_dict.update({'loss_total': loss})
        return loss, loss_dict


    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    @torch.no_grad()
    def log_images(self, batch, N = 4, n_row=4, sample=True, ddim_steps=200, ddim_eta=1.,
                  plot_diffusion_rows=False, plot_denoise_rows=False):
        use_ddim = ddim_steps is not None

        log = dict()

        x = batch['prim_rgba'].reshape(-1, self.bodydecoder.n_prims_y, self.bodydecoder.n_prims_x, 4, self.bodydecoder.prim_size, self.bodydecoder.prim_size, self.bodydecoder.prim_size) # supposed to be Bx64x64x16^3x4
        prim_scale = batch['delta_scale'].reshape(-1, self.bodydecoder.n_prims_y, self.bodydecoder.n_prims_x, 3)
        prim_scale = prim_scale[..., None, None, None].repeat(1, 1, 1, 1, self.bodydecoder.prim_size, self.bodydecoder.prim_size, self.bodydecoder.prim_size) / 20. * 255.
        x = torch.cat([x, prim_scale], dim=3) #supposed to be Bx32x32x7x8x8x8
        if self.dims == 2:
            x = x.permute(0, 4, 3, 1, 5, 2, 6).contiguous().reshape(-1, self.bodydecoder.prim_size*(4+3), self.bodydecoder.n_prims_y * self.bodydecoder.prim_size, self.bodydecoder.n_prims_x * self.bodydecoder.prim_size)
        else:
            raise NotImplementedError('dims {} not supported'.format(self.dims))
        x =  (x / 255) * 2 - 1
        x = x * self.scaling_factor

        c = None if self.d_cond is None else rearrange(batch['emb_conv'], 'b c h w -> b (h w) c').contiguous()

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)

        if plot_diffusion_rows:
            diffusion_row = list()
            z_start = x[:n_row]
            diffusion_row.append(z_start)
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    # print(z_noisy.shape)
                    # z_noisy should be rendered via rm
                    diffusion_row.append(z_noisy)
            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            log["diffusion_row"] = (diffusion_row / self.scaling_factor + 1) / 2. * 255.

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta)
            # (x / 255) * 2 - 1
            log["samples"] = (samples / self.scaling_factor + 1) / 2. * 255.
            if plot_denoise_rows:
                log["denoise_row"] = (torch.stack(z_denoise_row) / self.scaling_factor + 1) / 2. * 255
        return log

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None, **kwargs):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
        
    def apply_model(self, x, t, c):
        x = x / torch.std(x, dim=(1, 2, 3), keepdim=True) if self.normalize_bystd else x
        return self.diffusion(x, t, c)

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class UNetModel(nn.Module):
    """
    ## U-Net model
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            n_res_blocks: int,
            attention_levels: List[int],
            channel_multipliers: List[int],
            n_heads: int,
            tf_layers: int = 1,
            d_cond = None,
            dims = 2,
            conv_resample = True,
            use_checkpoint = True,
            scaling_factor = None,
            normalize_bystd = False,
            ):
        """
        :param in_channels: is the number of channels in the input feature map
        :param out_channels: is the number of channels in the output feature map
        :param channels: is the base channel count for the model
        :param n_res_blocks: number of residual blocks at each level
        :param attention_levels: are the levels at which attention should be performed
        :param channel_multipliers: are the multiplicative factors for number of channels for each level
        :param n_heads: the number of attention heads in the transformers
        """
        super().__init__()
        self.channels = channels

        # Number of levels
        levels = len(channel_multipliers)
        # Size time embeddings
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # Initial $3 \times 3$ convolution that maps the input to `channels`.
        # The blocks are wrapped in `TimestepEmbedSequential` module because
        # different modules have different forward function signatures;
        # for example, convolution only accepts the feature map and
        # residual blocks accept the feature map and time embedding.
        # `TimestepEmbedSequential` calls them accordingly.
        self.input_blocks.append(TimestepEmbedSequential(
            conv_nd(dims, in_channels, channels, 3, padding=1)))
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i], dims=dims, use_checkpoint=use_checkpoint)]
                channels = channels_list[i]
                dim_head = channels // n_heads
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(dims, channels, n_heads, dim_head, depth=tf_layers, context_dim=d_cond, use_checkpoint=use_checkpoint))
                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(channels, conv_resample, dims=dims)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        dim_head = channels // n_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb, dims=dims, use_checkpoint=use_checkpoint),
            SpatialTransformer(dims, channels, n_heads, dim_head, depth=tf_layers, context_dim=d_cond, use_checkpoint=use_checkpoint),
            ResBlock(channels, d_time_emb, dims=dims, use_checkpoint=use_checkpoint),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i], dims=dims, use_checkpoint=use_checkpoint)]
                channels = channels_list[i]
                dim_head = channels // n_heads
                # Add transformer
                if i in attention_levels:
                    layers.append(SpatialTransformer(dims, channels, n_heads, dim_head, depth=tf_layers, context_dim=d_cond, use_checkpoint=use_checkpoint))
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers.append(Upsample(channels, conv_resample, dims=dims))
                # Add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            zero_module(conv_nd(dims, channels, out_channels, 3, padding=1)),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """
        # To store the input half outputs for skip connections
        x_input_block = []

        # Get time step embeddings
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)
        # Output half of the U-Net
        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        # Final normalization and $3 \times 3$ convolution
        return self.out(x)


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs
    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            raise NotImplementedError("Downsample layer without use_conv!")

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0.,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, t_emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, t_emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        :param t_emb: is the time step embeddings of shape `[batch_size, d_t_emb]`
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(t_emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization
    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)




# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def max_neg_value(t):
    return -torch.finfo(t.dtype).max

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        if ATTENTION_MODE == 'xformers':
            q, k, v = map(lambda t: rearrange(t, '(b h) n d -> b n h d', h=h), (q, k, v))
            x = xformers.ops.memory_efficient_attention(q, k, v)
            out = rearrange(x, 'b n h d -> b n (h d)', h=h)
        elif ATTENTION_MODE == 'math':
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        else:
            raise NotImplementedError

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, use_checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None):
        # FIXME: hack for context here
        context = x if context is None else context
        return checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, dims, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        self.dims = dims
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = conv_nd(dims, in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, use_checkpoint=use_checkpoint)
                for d in range(depth)]
        )

        self.proj_out = zero_module(conv_nd(dims, inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.dims == 2:
            b, c, h, w = x.shape
        elif self.dims == 3:
            b, c, h, w, l = x.shape
        else:
            raise NotImplementedError('dimension {} not supported!'.format(self.dims))
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.dims == 2:
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        elif self.dims == 3:
            x = rearrange(x, 'b c h w l -> b (h w l) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if self.dims == 2:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        elif self.dims == 3:
            x = rearrange(x, 'b (h w l) c -> b c h w l', h=h, w=w, l=l).contiguous()
        x = self.proj_out(x)
        return x + x_in


"""SAMPLING ONLY."""

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        e_t = self.model.apply_model(x, t, c)

        if unconditional_guidance_scale != 1.:
            assert unconditional_conditioning is not None
            e_t_uncond =  self.model.apply_model(x, t, unconditional_conditioning)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization == "eps":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        elif self.model.parameterization == 'x0':
            pred_x0 = e_t
            e_t = self.model.predict_noise_from_start(x, t, pred_x0)
        else:
            raise NotImplementedError()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
