import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dva.geom import (
    make_postex,
    compute_tbn,
    axisangle_to_matrix,
    project_points_multi,
    GeometryModule,
)
import dva.layers as la
from dva.layers import ConvBlock, tile2d
from primdiffusion.model.transformer import SpatialTransformer
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection
from pytorch3d.structures import Meshes

from easymocap.smplmodel import SMPLlayer

import logging

logger = logging.getLogger(__name__)


def init_primitives(slab_size, n_prims, lbs_fn, geo_fn, ref_frame, scale=15000.0):
    stride = slab_size // int(n_prims**0.5)
    device = geo_fn.vt.device
    _, face_index_imp, bary_index_imp = geo_fn.render_index_images(
        slab_size, impaint=True
    )

    bary_index_imp = th.as_tensor(bary_index_imp, device=device)

    prim_bary_img = bary_index_imp[stride // 2 :: stride, stride // 2 :: stride]
    prim_vidx_img = geo_fn.vi[
        face_index_imp[stride // 2 :: stride, stride // 2 :: stride]
    ]
    prim_vtidx_img = geo_fn.vti[
        face_index_imp[stride // 2 :: stride, stride // 2 :: stride]
    ]

    # getting actual geometrical coordinates
    ref_frame = {
        "poses": th.as_tensor(ref_frame["poses"]),
        "shapes": th.as_tensor(ref_frame["shapes"]),
        "Rh": th.as_tensor(ref_frame["Rh"]),
        "Th": th.as_tensor(ref_frame["Th"]),
    }

    # convert to mm
    geom = lbs_fn(**ref_frame) * 1000.0

    prim_pos_mesh = (
        make_postex(geom, prim_vidx_img, prim_bary_img)
        .permute(0, 2, 3, 1)
        .reshape(n_prims, 3)
    )
    distance = th.cdist(prim_pos_mesh, prim_pos_mesh)

    # get a small neigbhourhood around
    nbs_dists = th.topk(distance, k=24, largest=False).values[:, 1:].mean(dim=-1)
    nbs_dists = nbs_dists.clip(5.0, 50.0)
    prim_scale = scale * (1.0 / nbs_dists)

    return prim_vidx_img, prim_vtidx_img, prim_bary_img, prim_scale, geom


class BodyDecoder(nn.Module):
    def __init__(
        self,
        assets,
        n_prims,
        prim_size,
        n_pose_dims,
        n_pose_enc_channels,
        n_embs_channels=64,
        prim_motion_enabled=False,
        prim_motion_start_train=100,
        prim_rt_enabled=True,
        n_init_channels=64,
        uv_size=512,
        smpl_gender="neutral",
        image_height=1024,
        image_width=1024,
    ):
        super().__init__()

        self.uv_size = uv_size

        self.lbs_fn = SMPLlayer(
            assets.smpl_path,
            model_type="smpl",
            gender=smpl_gender,
        )

        # initializing primitives
        self.n_prims = n_prims
        self.n_prims_x = int(n_prims**0.5)
        self.n_prims_y = int(n_prims**0.5)
        self.prim_size = prim_size
        self.slab_size = int(n_prims**0.5 * prim_size)

        logger.info(
            f"slab_size={self.slab_size}, prim_size={self.prim_size}, n_prims={self.n_prims}"
        )

        self.prim_motion_enabled = prim_motion_enabled
        self.prim_motion_start_train = prim_motion_start_train
        self.prim_rt_enabled = prim_rt_enabled

        logger.info("initializing geometry module...")
        self.geo_fn = GeometryModule(
            th.as_tensor(assets.topology.vi, dtype=th.long),
            th.as_tensor(assets.topology.vt, dtype=th.float32),
            th.as_tensor(assets.topology.vti, dtype=th.long),
            th.as_tensor(assets.topology.v2uv, dtype=th.long),
            impaint=True,
            uv_size=uv_size,
        )

        logger.info("done!")

        logger.info("initiailizing primitives")
        prim_vidx_img, prim_vtidx_img, prim_bary_img, prim_scale, ref_geom = init_primitives(
            self.slab_size,
            self.n_prims,
            self.lbs_fn,
            self.geo_fn,
            assets["ref_frame"],
        )
        logger.info("done!")

        self.register_buffer("prim_vidx_img", prim_vidx_img, persistent=False)
        self.register_buffer("prim_vtidx_img", prim_vtidx_img, persistent=False)
        self.register_buffer("prim_bary_img", prim_bary_img, persistent=False)
        self.register_buffer("prim_scale", prim_scale, persistent=False)
        self.register_buffer("ref_geom", ref_geom, persistent=False)

        # rasterization with pytorch3d
        self.image_height = image_height
        self.image_width = image_width
        self.rasterizer = MeshRasterizer(
            raster_settings=RasterizationSettings(
                image_size=(image_height, image_width),
            )
        )

        # Module compressing pose to localized
        self.local_pose_conv_block = ConvBlock(
            n_pose_dims,
            n_pose_enc_channels,
            self.n_prims_x,
            kernel_size=1,
            padding=0,
        )

        # Finally aggregate everything
        self.joint_conv_block = ConvBlock(
            n_pose_enc_channels + n_embs_channels,
            n_init_channels,
            self.n_prims_x,
        )

        S = uv_size
        self.embs_encoder = nn.Sequential(
            nn.UpsamplingBilinear2d(size=(S // 2, S // 2)),
            la.Conv2dWNUB(3, 4, S // 4, S // 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(4, 8, S // 8, S // 8, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(8, 16, S // 16, S // 16, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.UNetWB(16, n_embs_channels, S // 16, out_scale=1.0),
        )
        self.apply(lambda x: la.weights_initializer(x, 0.2))

        # primitive motion decoder
        self.decoder_motion = DeconvMotionDecoder(n_prims, n_init_channels)

        # Module decoding RGB texture map
        self.decoder_rgb = DeconvContentDecoder(
            n_prims, prim_size, n_init_channels, 3
        )
        # Module decoding Alpha texture map
        self.decoder_alpha = DeconvContentDecoder(
            n_prims, prim_size, n_init_channels, 1
        )
        # Geometry decoder
        self.decoder_geom = LocalDeconvDecoder(n_prims, n_init_channels, 3)
        self.relu = th.nn.ReLU()

        # cross-modal attention
        self.transformer = SpatialTransformer(n_pose_enc_channels + n_embs_channels, 4, 64, 2)

    def _compute_cond_tex(
        self, geom, cond_Rt, cond_K, cond_image, vis_depth_threshold=20.0
    ):
        B, NC = cond_Rt.shape[:2]
        H, W = self.image_height, self.image_width
        S = self.uv_size
        device = cond_Rt.device

        # computing aggreagated texture from conditioning images
        cond_Rt_flat = cond_Rt.reshape(B * NC, 3, 4)
        cond_K_flat = cond_K.reshape(B * NC, 3, 3)
        cond_image_flat = cond_image.reshape(B * NC, 3, H, W)
        # converting to p3d format
        image_size_tensor = th.as_tensor(
            [[H, W]], dtype=th.float32, device=device
        ).expand(B * NC, -1)

        cond_cameras = cameras_from_opencv_projection(
            R=cond_Rt_flat[:, :3, :3],
            tvec=cond_Rt_flat[:, :3, -1],
            camera_matrix=cond_K_flat,
            image_size=image_size_tensor,
        )

        # repeating geometry
        cond_geom = geom.repeat_interleave(NC, dim=0)
        cond_faces = self.geo_fn.vi[np.newaxis].expand(B * NC, -1, -1)

        # rasterizing depth (we need that to compute depth)
        meshes = Meshes(cond_geom, cond_faces)

        fragments = self.rasterizer(
            meshes,
            cameras=cond_cameras,
        )

        geom_uv = self.geo_fn.to_uv(geom).reshape(B, 3, -1).permute(0, 2, 1)

        # *projected* depth
        pix_uv, depth_uv = project_points_multi(
            geom_uv, cond_Rt, cond_K, normalize=True, size=(H, W)
        )

        pix_uv = pix_uv.reshape(B * NC, S, S, 2)
        depth_uv = depth_uv.reshape(B * NC, 1, S, S)

        # *rendered* depth
        raster_depth = fragments.zbuf[:, np.newaxis, ..., 0]
        raster_depth_uv = F.grid_sample(
            raster_depth, pix_uv, mode="bilinear", align_corners=False
        )

        vis_mask_uv = (
            th.norm(depth_uv - raster_depth_uv, dim=1, keepdim=True)
            < vis_depth_threshold
        ).float()

        cond_tex = (
            F.grid_sample(cond_image_flat, pix_uv, mode="bilinear", align_corners=False)
            * vis_mask_uv
        ).reshape(B, NC, 3, S, S)
        vis_mask_uv = vis_mask_uv.reshape(B, NC, 1, S, S)

        cond_tex_agg = (cond_tex * vis_mask_uv).sum(dim=1) / (
            vis_mask_uv.sum(dim=1) + 1.0e-6
        )

        return cond_tex_agg

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
        B = poses.shape[0]

        # first 3 numbers encode global pose
        local_pose = poses[:, 3:]

        verts_lbs = self.lbs_fn(
            poses=poses,
            shapes=shapes,
            Rh=Rh,
            Th=Th,
            v_template=self.lbs_fn.v_template[np.newaxis],
        )

        # converting to mm
        geom = verts_lbs * 1000.0

        cond_tex = self._compute_cond_tex(self.ref_geom.repeat(B, 1, 1), cond_Rt, cond_K, cond_image)
        embs_conv = self.embs_encoder(cond_tex / 255.0)

        pose_masked = tile2d(local_pose, self.n_prims_x)
        pose_conv = self.local_pose_conv_block(pose_masked)

        joint = th.cat([pose_conv, embs_conv], dim=1)
        joint = self.transformer(joint)
        joint = self.joint_conv_block(joint)

        prim_pos_mesh = (
            make_postex(geom, self.prim_vidx_img, self.prim_bary_img)
            .permute(0, 2, 3, 1)
            .reshape(-1, self.n_prims, 3)
            .detach()
        )

        prim_scale_mesh = (
            self.prim_scale[np.newaxis, :, np.newaxis].expand(B, -1, 3).detach()
        )

        tbn = compute_tbn(geom, self.geo_fn.vt, self.prim_vidx_img, self.prim_vtidx_img)

        prim_rot_mesh = (
            th.stack(tbn, dim=-2)
            .reshape(B, self.n_prims, 3, 3)
            .permute(0, 1, 3, 2)
            .contiguous()
            .detach()
        )

        # correctives
        delta_pos, delta_rvec, delta_scale = self.decoder_motion(joint)

        # if needed, do not train correctives
        if not self.prim_motion_enabled or train_iter < self.prim_motion_start_train:
            delta_pos = delta_pos * 0.0
            delta_rvec = delta_rvec * 0.0
            delta_scale = delta_scale * 0.0 + 1.0
        
        if not self.prim_rt_enabled:
            delta_pos = delta_pos * 0.0
            delta_rvec = delta_rvec * 0.0

        prim_pos = prim_pos_mesh.detach() + delta_pos
        prim_scale = prim_scale_mesh.detach() * delta_scale

        primrotdelta = axisangle_to_matrix(delta_rvec)
        prim_rot = th.bmm(
            prim_rot_mesh.detach().view(-1, 3, 3), primrotdelta.view(-1, 3, 3)
        ).reshape(B, self.n_prims, 3, 3)

        view_cond = joint
        prim_rgb_texture = (self.decoder_rgb(view_cond)).view(
            B,
            self.prim_size,
            3,
            self.n_prims_y * self.prim_size,
            self.n_prims_x * self.prim_size,
        )
        prim_rgb_texture = F.relu(25.0 * prim_rgb_texture + 100.0).clip(0.0, 255.0)

        prim_alpha_texture = self.decoder_alpha(joint).view(
            B,
            self.prim_size,
            1,
            self.n_prims_y * self.prim_size,
            self.n_prims_x * self.prim_size,
        )

        prim_alpha_texture = F.relu(25.0 * prim_alpha_texture + 100.0).clip(0.0, 255.0)

        prim_rgba = th.cat((prim_rgb_texture, prim_alpha_texture), 2).view(
            B,
            self.prim_size,
            4,
            self.n_prims_y,
            self.prim_size,
            self.n_prims_x,
            self.prim_size,
        )
        prim_rgba = prim_rgba.permute(0, 3, 5, 2, 1, 4, 6)
        prim_rgba = prim_rgba.reshape(
            B, self.n_prims, 4, self.prim_size, self.prim_size, self.prim_size
        )

        return dict(
            prim_rgba=prim_rgba,
            prim_pos=prim_pos,
            prim_rot=prim_rot,
            prim_scale=prim_scale,
            verts=geom,
            texture_rgb=prim_rgb_texture,
            texture_alpha=prim_alpha_texture,
            delta_scale=delta_scale,
            delta_pos=delta_pos,
            cond_tex=cond_tex,
            emb_conv=embs_conv,
            pose_conv=pose_conv,
            prim_mesh_scale=prim_scale_mesh.detach(),
        )


class DeconvMotionDecoder(nn.Module):
    def __init__(self, n_prims, in_channels):
        super().__init__()
        self.n_prims = n_prims
        self.wh = int(n_prims ** 0.5)
        self.dec_conv = nn.Sequential(
            la.Conv2dWNUB(in_channels, 64, self.wh, self.wh, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(64, 128, self.wh, self.wh, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(128, 64, self.wh, self.wh, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(64, 64, self.wh, self.wh, 3, 1, 1),
            nn.LeakyReLU(0.2),
            la.Conv2dWNUB(64, 9, self.wh, self.wh, 3, 1, 1),
        )
        self.apply(lambda x: la.weights_initializer(x, 0.2))
        la.weights_initializer(self.dec_conv[-1])

    def forward(self, ftrs):
        out = self.dec_conv(ftrs)
        out = out.view(ftrs.size(0), 9, -1).permute(0, 2, 1).contiguous()
        #
        primposdelta = out[:, :, 0:3] * 0.1
        primrvecdelta = out[:, :, 3:6] * 0.01
        primscaledelta = th.exp(0.01 * out[:, :, 6:9])
        return primposdelta, primrvecdelta, primscaledelta


class DeconvContentDecoder(nn.Module):
    def __init__(self, n_prims, primsize, inch, outch):
        super().__init__()
        self.n_prims = n_prims
        self.primsize_w = primsize
        self.primsize_h = primsize
        self.primsize_d = primsize
        self.outch = outch
        nh = int(np.sqrt(self.n_prims))
        nw = nh
        assert nw * nh == self.n_prims

        self.nh = nh
        self.conv_net = nn.Sequential(
            la.ConvTranspose2dWNUB(inch, 32, nh*2, nh*2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(32, 16, nh*4, nh*4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(16, self.primsize_d * self.outch, nh*8, nh*8, 4, 2, 1),
        )
        self.apply(lambda x: la.weights_initializer(x, 0.2))
        la.weights_initializer(self.conv_net[-1])

    def forward(self, ftrs: th.Tensor) -> th.Tensor:
        return self.conv_net(ftrs)


class LocalDeconvDecoder(nn.Module):
    def __init__(self, n_prims, in_channels, out_channels):
        super().__init__()
        self.n_prims = n_prims
        self.out_channels = out_channels

        nh = int(np.sqrt(self.n_prims))
        nw = nh
        assert nw * nh == self.n_prims
        self.nh = nh

        self.conv_net = nn.Sequential(
            la.ConvTranspose2dWNUB(in_channels, 128, 128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(128, 64, 256, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(64, 32, 512, 512, 4, 2, 1),
            nn.LeakyReLU(0.2),
            la.ConvTranspose2dWNUB(32, out_channels, 1024, 1024, 4, 2, 1),
        )
        self.apply(lambda x: la.weights_initializer(x, 0.2))
        la.weights_initializer(self.conv_net[-1], 1.0)

    def forward(self, ftrs: th.Tensor) -> th.Tensor:
        return self.conv_net(ftrs)
