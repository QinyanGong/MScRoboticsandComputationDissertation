# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import os
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import PIL
from PIL import Image
import torch
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat

from nerfstudio.model_components.gs_sdf_loss import GS_SDF_Loss

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE



from diffusers import DDIMScheduler,UNet2DModel,DDIMPipeline

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat



@dataclass
class DDIMSplatfactoModelConfig(SplatfactoModelConfig):
    """DDIM Splatfacto Model Config, extending the Splatfacto Model Config for DDIM specifics"""
    _target: Type = field(default_factory=lambda: DDIMSplatfactoModel)
    gaussian_pertub_level: float = 0.01
    """scale of noise pertubing gaussians' means"""
    ddim_scheduler_path: str = "/root/ddim/DDIM/scheduler"
    """scheduler for pretrained DDIM model"""
    unet_model_path: str ="/root/ddim/DDIM/unet"
    """pretrained unet model"""
    timesteps: torch.LongTensor = torch.LongTensor([990])
    "timesteps used for diffusion scheduler"
    # Resize to 256x256
    target_size: tuple = (256, 256)
    "target size for diffusion model training"
    empty_lambda: float = 1e4
    "weight for empty loss that penalize on gaussians apparing in empty space"
    psnr_lambda: float = 0.02
    """weight of psnr loss"""
    sdf_lambda: float = 1e-2
    """weight of sdf loss"""
    df_img_output_dir: str = "/root/renders/cecum_t1_a/df_gt"
    """diffusion image output directory"""
    diffusion_lambda = 0.002
    """learning rate on diffusion images"""
    output_depth_during_training: bool = True
    """render depth in the process"""
    lpips_lambda = 0.2
    """weight of lpips"""

class DDIMSplatfactoModel(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting with DDIM specifics

    Args:
        config: DDIMSplatfacto configuration to instantiate model
    """

    config: DDIMSplatfactoModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)
        self.df_idx = 0
        """index to save diffusion images"""
        self.sdfloss = 0

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            # Generates a tensor with random values uniformly distributed in the interval [0, 1).
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        # check if seed points are provided and valid
        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            # self.seed_points[1].shape[0]: Number of seed points.
            # dim_sh: the number of spherical harmonics bases
            # RGB channels
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                # Converts RGB colors to spherical harmonics (SH) coefficients using RGB2SH.
                # Assigns SH coefficients to the zero degree (direct component).
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                # Sets higher-degree coefficients to zero.
                shs[:, 1:, 3:] = 0.0
            else:
                # color-only optimization
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            # Direct component of SH and higher-degree component of SH
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        # torch.logit: Applies the logit function, 
        # which is the inverse of the sigmoid function. 
        # This maps the values from the range [0, 1] to (-inf, inf).
        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        # torch.nn.ParameterDict: A dictionary-like container for torch.nn.Parameter objects. 
        # This allows for easy management and access to the model parameters.
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )
        # self.num_train_data: number of cameras 
        # return a CameraOptimizer object
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        # The metrics (PSNR, SSIM, LPIPS) are crucial for assessing the quality of the generated images. 
        # PeakSignalNoiseRatio: Measures the ratio between 
        #       the maximum possible power of a signal and the power of corrupting noise. 
        #       Initialized with a data_range of 1.0.
        # SSIM (Structural Similarity Index): Measures the similarity between two images. 
        #       Initialized with data_range=1.0, size_average=True, and channel=3 for RGB images.
        # LearnedPerceptualImagePatchSimilarity: Measures perceptual similarity using deep learning-based methods. 
        #       Normalized to ensure consistent scales.
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        # self.crop_box: Initializes the crop box as None, 
        #       which could be used later for defining a region of interest.
        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.noisy_gaussian()


    # The @property: define getter methods in a class
    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    
    def noise_like(self,shape, device="cuda", repeat=False):
        """Generate means of a common source of noise with the same shape as provided"""
        repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
        noise = lambda: torch.randn(shape, device=device)
        return repeat_noise() if repeat else noise()
    
    def noisy_gaussian(self):
        """Generate noisy 3d gaussians as noise source"""
        means = self.noise_like(self.means.shape, device="cuda", repeat=False)
        max_values = torch.max(self.gauss_params["means"], dim=0)[0].to("cuda")
        # Multiply the first 3 elements of means by the corresponding max values
        means[:3] *= max_values[:3]
        dim_sh = num_sh_bases(self.config.sh_degree)
        features_dc = torch.rand(self.num_points, 3, device="cuda")
        features_rest = torch.randn((self.num_points, dim_sh - 1, 3), device= "cuda")
        # opacities normally distributed around 0.1
        opacities = torch.logit(0.1 * (torch.rand(self.num_points, 1, device="cuda")/2+0.5))
        self.noisy_gaussians = {
            "means": means,
            "scales": self.scales,
            "quats": self.quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
        }

    
    def noisy_image(self, camera: Cameras):
        if not isinstance(camera, Cameras):
            print("Called get_noisy_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.noisy_gaussians["means"]).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.noisy_gaussians["opacities"][crop_ids]
            means_crop = self.noisy_gaussians["means"][crop_ids]
            features_dc_crop = self.noisy_gaussians["features_dc"][crop_ids]
            features_rest_crop = self.noisy_gaussians["features_rest"][crop_ids]
            scales_crop = self.noisy_gaussians["scales"][crop_ids]
            quats_crop = self.noisy_gaussians["quats"][crop_ids]
        else:
            opacities_crop = self.noisy_gaussians["opacities"]
            means_crop = self.noisy_gaussians["means"]
            features_dc_crop = self.noisy_gaussians["features_dc"]
            features_rest_crop = self.noisy_gaussians["features_rest"]
            scales_crop = self.noisy_gaussians["scales"]
            quats_crop = self.noisy_gaussians["quats"]

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # both RGB and depth (ED stands for depth) information will be included in the output.
            render_mode = "RGB+ED" 
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop) # [0,1]
            sh_degree_to_use = None

        # Pertub the Gaussians' means for Variational Gaussian Splatting
        gaussian_mean_noise = (torch.randn(means_crop.shape[0], 3) * self.config.gaussian_pertub_level).to("cuda")
        # resterization from gsplat.rendering 
        render, alpha, info = rasterization(
            means=means_crop + gaussian_mean_noise, # pertub means with noise
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        alpha = alpha[:, ...]

        # Preparing Background and Compositing RGB
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # if render_mode == "RGB+ED":
        #     depth_im = render[:, ..., 3:4]
        #     depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        # else:
        #     depth_im = None

        # # Checks if the background color has 3 channels (RGB) and if the model is not in training mode.
        # if background.shape[0] == 3 and not self.training:
        #     background = background.expand(H, W, 3)

        # {    "depth": depth_im,  
                #     "accumulation": alpha.squeeze(0),  
                #     "background": background,  
                # }  

        return rgb.squeeze(0)        



    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0] # e.g. 200 number of points
        for name, param in self.gauss_params.items():
            old_shape = param.shape  # e.g. (100,64,64,3)
            new_shape = (newp,) + old_shape[1:]  # e.g. (200,) + # (,64,64,3) = (200,64,64,3)
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        # super(): Refers to the parent class (torch.nn.Module) from which this model class inherits.
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        # x.cpu().numpy(): Moves the tensor to the CPU and converts it to a NumPy array. 
        # This is necessary because scikit-learn works with NumPy arrays, not PyTorch tensors.
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        # n_neighbors=k + 1: Finds k+1 neighbors because the nearest neighbor of a point is the point itself.
        # algorithm="auto": Chooses the best algorithm based on the input data.
        # metric="euclidean": Uses Euclidean distance to measure similarity.
        # fit(x_np): Fits the nearest neighbors model to the input data.
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors,returning distances and indices.
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return: distances[:, 1:]
        # astype(np.float32): Converts the distances and indices to float32 for consistency with PyTorch tensor types
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        # Retrieves the first parameter from the optimizer's parameter group.
        param = optimizer.param_groups[0]["params"][0]
        # Fetches the optimizer state for this parameter 
        # and then deletes the parameter from the optimizer's state dictionary.
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        # Uses the deleted_mask to filter out specific entries from these states, 
        # effectively removing them.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        # get_gaussian_param_groups: This method is assumed to return a dictionary 
        # where keys are group names and values are parameter lists. 
        # These groups categorize different sets of Gaussian parameters in the model.
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        #  Frees up unused memory in the CUDA cache, 
        # which can help in managing GPU memory more efficiently, 
        # especially after removing parameters.
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        # e.g. param_state['exp_avg'] = torch.tensor([
        # [0.1, 0.2, 0.3],
        # [1.1, 1.2, 1.3],
        # [2.1, 2.2, 2.3],
        # [3.1, 3.2, 3.3],
        # [4.1, 4.2, 4.3]
        # ])
        # dup_mask = torch.tensor([True, False, True, False, False])
        # elements_to_duplicate = param_state["exp_avg"][dup_mask.squeeze()]
        # elements_to_duplicate:
        # tensor([
        #   [0.1, 0.2, 0.3],
        #   [2.1, 2.2, 2.3]
        # ])

        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            # repeat_dims = (2, 1)
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    # *repeat_dims: * unpacks the tuple into separate arguments
                    # zeros_like_elements:
                    # tensor([
                    #   [0.0, 0.0, 0.0],
                    #   [0.0, 0.0, 0.0],
                    #   [0.0, 0.0, 0.0],
                    #   [0.0, 0.0, 0.0]
                    # ])
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        # updates after each training step, specifically focusing on gradient norms and the maximum screen size of points.
        # Ensures the current step matches the expected step stored in self.step
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            # Identifies visible points where radii are greater than zero.
            visible_mask = (self.radii > 0).flatten()
            # Calculates the norm of the gradients for visible points along the last dimension. 
            # The dim=-1 argument specifies that the norm should be computed across the last dimension of the tensor, 
            # which typically represents the feature dimension.
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            # If xys_grad_norm is not initialized, creates tensors for gradient norms and visibility counts.
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)
            assert self.vis_counts is not None
            # Update Counts and Norms: Increments visibility counts and adds current gradients to the moving average.
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max visible point size, as a ratio of number of pixels

            # self.radii: A tensor containing the radii of points.
            # visible_mask: A boolean tensor indicating which points are visible.
            # self.max_2Dsize: A tensor storing the maximum 2D size observed for each point.
            # self.last_size: A tuple or list containing the dimensions of the last rendered frame.

            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            # self.radii.detach(): Creates a detached tensor (i.e., no gradient tracking) from self.radii.
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                # self.step: The current training step.
                # self.config.stop_split_at: A configuration parameter that indicates 
                #     the training step after which no further splitting should occur. 
                self.step < self.config.stop_split_at
                # This calculates the remainder when the current step is divided by the reset interval, 
                # giving the position within the interval.
                # Ensures that densification only occurs if the current position within the reset interval 
                # is beyond the number of training data points plus the refinement interval. 
                # This avoids performing densification immediately after a reset, giving the model time to stabilize.
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            # The densification process aims to refine the model 
            # by splitting and duplicating Gaussian parameters 
            # based on certain conditions and thresholds.
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                # Average Gradient Norm Calculation: Computes the average gradient norm for each point. 
                # It scales the gradient norms by 0.5 and the maximum of the last frame's dimensions.
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                # Identifies points with gradient norms higher than a specified threshold (densify_grad_thresh).
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                # Size Threshold: Selects points with scales larger than densify_size_thresh.
                # self.scales.exp().max(dim=-1).values: This extracts the maximum value of the exponential of the scales along the last dimension.
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    # splits |= ...: Performs an element-wise logical OR operation. 
                    # This updates splits so that any point that satisfies either 
                    # the original size condition or the screen size condition will be marked for splitting.
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                # This updates splits so that only points that satisfy both the updated size condition 
                # and the high gradient condition will be marked for splitting
                splits &= high_grads
                nsamps = self.config.n_split_samples # split into 2 samples 
                split_params = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )
                # torch.where(splits)[0]: This finds the indices of the True elements in the splits boolean tensor.
                # If splits is [True, False, True], torch.where(splits)[0] will return [0, 2].
                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                # Creates a mask indicating which points were split.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at \
                and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                # torch.clamp(..., max=...): Clamps the opacity values 
                # to ensure they do not exceed the calculated maximum value.
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    # torch.tensor(reset_value, device=self.device): 
                    #       Converts reset_value to a tensor on the specified device.
                    # torch.logit(...): Applies the logit function 
                    #       (inverse of the sigmoid function) to the reset_value tensor
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                # Reset Exponential Moving Averages
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize is not None:
                    toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups


    # the method returns a list where each element is an instance of the TrainingCallback class.
    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = [] #An empty list that will store the training callbacks
        # [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION]: Specifies the location where this callback should be executed.
        # self.step_cb: The callback function to be executed.
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                # Additional arguments passed to the callback function, in this case, the optimizers.
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:          
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1


    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image


    # The @staticmethod decorator in Python is used to define a method that belongs to a class 
    # but does not require access to any instance-specific data or methods. 
    # This means that the method can be called on the class itself, without creating an instance of the class.
    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    # This method ensures that the outputs are properly initialized and 
    # ready for further processing in an image or video processing pipeline.
    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_noisy_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_noisy_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # both RGB and depth (ED stands for depth) information will be included in the output.
            render_mode = "RGB+ED" 
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop) # [0,1]
            sh_degree_to_use = None

        # # Pertub the Gaussians' means for Variational Gaussian Splatting
        # gaussian_mean_noise = (torch.rand(self.num_points, 3)-0.5) * self.config.gaussian_pertub_level
        # gaussian_mean_noise = gaussian_mean_noise.to(self.device)
        # resterization from gsplat.rendering 
        render, alpha, info = rasterization(
            means=means_crop, #+ gaussian_mean_noise, # pertub means with noise
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        alpha = alpha[:, ...]

        # Preparing Background and Compositing RGB
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        # Checks if the background color has 3 channels (RGB) and if the model is not in training mode.
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0), 
            "depth": depth_im,  
            "accumulation": alpha.squeeze(0),  
            "background": background,  
        }  
    
    def get_df_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        splat_output = self.get_noisy_outputs(camera)
        rgb_output = splat_output["rgb"]
        output_shape = rgb_output.shape[:2]
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_output = torch.moveaxis(rgb_output, -1, 0)[None, ...]
        
        # resize to [1,3,256,256]
        rgb_output = F.interpolate(rgb_output, size=self.config.target_size, mode='bilinear', align_corners=False)

        # # novel image path
        # novel_img_path = "/root/renders/cecum_t1_a/novel_img"

        # # Save the novel images
        # if self.config.df_img_output_dir is not None:
        #     os.makedirs(novel_img_path, exist_ok=True)
        #     # Define the file name and path
        #     image_name = f"{self.df_idx}_df.png"
        #     file_path = os.path.join(novel_img_path, image_name)

        #     # Save the image using torchvision's save_image
        #     vutils.save_image(rgb_output[0], file_path)
        #     self.df_idx += 1
        # # print(rgb_output[:2])

        # Normalize the rgb values from [0, 1] to [-1, 1]
        normalize_rgb = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        rgb_output = normalize_rgb(rgb_output)
        # print(rgb_output[:2])

        # Add noise to model output using DDIM noise scheduler
        noise = torch.randn(rgb_output.shape).to("cuda")
        # structured_noise = self.noisy_image(camera=camera).to(device="cpu")
        # # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        # structured_noise = torch.moveaxis(structured_noise, -1, 0)[None, ...]
        # # resize to [1,3,256,256]
        # structured_noise = F.interpolate(structured_noise, size=self.config.target_size, mode='bilinear', align_corners=False)
        
        # noise image path
        # noise_img_path = "/root/renders/cecum_t1_a/noise_img"
        # # Save the novel images
        # if self.config.df_img_output_dir is not None:
        #     os.makedirs(noise_img_path, exist_ok=True)
        #     # Define the file name and path
        #     image_name = f"{self.df_idx}_df.png"
        #     file_path = os.path.join(noise_img_path, image_name)

        #     # Save the image using torchvision's save_image
        #     # Optionally, scale the image from [0, 1] to [0, 255] using the `normalize` argument if needed
        #     vutils.save_image(structured_noise, file_path)
        #     self.df_idx += 1

        # # normalize
        # structured_noise = normalize_rgb(structured_noise)
        # noise = structured_noise+noise

        noise_scheduler = DDIMScheduler.from_pretrained(self.config.ddim_scheduler_path)
        trained_diffusion_model = UNet2DModel.from_pretrained(self.config.unet_model_path).to("cuda")

        noisy_rgb_output = noise_scheduler.add_noise(rgb_output, noise, self.config.timesteps.to("cuda"))
        pre_image = noisy_rgb_output

        # # noisy image path
        # noisy_img_path = "/root/renders/cecum_t1_a/noisy_img"
        # # Save the novel images
        # if self.config.df_img_output_dir is not None:
        #     os.makedirs(noisy_img_path, exist_ok=True)
        #     # Define the file name and path
        #     image_name = f"{self.df_idx}_df.png"
        #     file_path = os.path.join(noisy_img_path, image_name)

        #     # Save the image using torchvision's save_image
        #     # Optionally, scale the image from [0, 1] to [0, 255] using the `normalize` argument if needed
        #     vutils.save_image(noisy_rgb_output[0], file_path)
        #     self.df_idx += 1

        noise_scheduler.set_timesteps(num_inference_steps=50)
        # new_image_pipe = DDIMPipeline.from_pretrained("./ddim/ddim-cecum_t1_a")
        # process noisy image with DDIM
        with torch.no_grad():
            for t in noise_scheduler.timesteps:
                # 1. predict noise model_output
                noise_output = trained_diffusion_model(pre_image, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                pre_output = noise_scheduler.step(
                    noise_output, t, pre_image
                )
                pre_image = pre_output.prev_sample

        # Denormalize the image after diffusion, from [-1, 1] to [0, 1]
        denormalize = transforms.Normalize([-1, -1, -1], [2, 2, 2])
        denormalized_image = denormalize(pre_image)

        # reshape from [1,3,256,256] to [1,3,H,W]
        denormalized_image = F.interpolate(denormalized_image, size=output_shape, mode='bilinear', align_corners=False)
        # Convert back from [1, C, H, W] to [H, W, C]
        splat_output["rgb"] = torch.moveaxis(denormalized_image[0].to("cuda"), 0, -1)

        # # Save the diffusion images
        # if self.config.df_img_output_dir is not None:
        #     os.makedirs(self.config.df_img_output_dir, exist_ok=True)
        #     # Define the file name and path
        #     image_name = f"{self.df_idx}_df.png"
        #     file_path = os.path.join(self.config.df_img_output_dir, image_name)

        #     # Save the image using torchvision's save_image
        #     # Optionally, scale the image from [0, 1] to [0, 255] using the `normalize` argument if needed
        #     vutils.save_image(denormalized_image[0].to("cuda"), file_path)
        #     self.df_idx += 1

        return splat_output
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        # Concatenates direct and rest features to form colors.
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # both RGB and depth (ED stands for depth) information will be included in the output.
            render_mode = "RGB+ED" 
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            # If self.config.sh_degree is 4 and self.config.sh_degree_interval is 1000, 
            # then at step 3000, sh_degree_to_use will be min(3000 // 1000, 4) = min(3, 4) = 3.
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop) # [0,1]
            sh_degree_to_use = None
        # resterization from gsplat.rendering 
        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        # If both conditions are true, it retains the gradients for means2d. 
        # This is useful for backpropagation, 
        # ensuring that gradient information is not lost during the training process.
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        # Preparing Background and Compositing RGB
        background = self._get_background_color()
        # render[:, ..., :3]: Extracts the RGB channels from the rendered image.
        # (1 - alpha) * background: Computes the contribution of the background based on the alpha values.
        rgb = render[:, ..., :3] + (1 - alpha) * background
        # Ensures that the RGB values are within the range [0, 1] using torch.clamp
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]

            # torch.where(condition, x, y):
            # condition: A boolean tensor that determines which elements to choose from x and y.
            # x: The tensor to select elements from where condition is True.
            # y: The tensor to select elements from where condition is False.
            # depth_im.detach().max(): Finds the maximum value in the detached depth_im tensor. 
            # This value is used to fill in the regions where alpha is zero.
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        # Checks if the background color has 3 channels (RGB) and if the model is not in training mode.
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        
        outputs ={
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

        # outputs["rgb"] = self.render_df(outputs["rgb"]).clamp(min=0, max=1)
        return outputs

    def render_df(self,rgb_output):
        output_shape = rgb_output.shape[:2]
        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_output = torch.moveaxis(rgb_output, -1, 0)[None, ...]
        
        # resize to [1,3,256,256]
        rgb_output = F.interpolate(rgb_output, size=self.config.target_size, mode='bilinear', align_corners=False)

        # Normalize the rgb values from [0, 1] to [-1, 1]
        normalize_rgb = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        rgb_output = normalize_rgb(rgb_output)
        # print(rgb_output[:2])

        # Add noise to model output using DDIM noise scheduler
        noise = torch.randn(rgb_output.shape).to("cuda")

        noise_scheduler = DDIMScheduler.from_pretrained(self.config.ddim_scheduler_path)
        trained_diffusion_model = UNet2DModel.from_pretrained(self.config.unet_model_path).to("cuda")

        noisy_rgb_output = noise_scheduler.add_noise(rgb_output, noise, self.config.timesteps.to("cuda"))
        pre_image = noisy_rgb_output

        noise_scheduler.set_timesteps(num_inference_steps=50)
        # new_image_pipe = DDIMPipeline.from_pretrained("./ddim/ddim-cecum_t1_a")
        # process noisy image with DDIM
        with torch.no_grad():
            for t in noise_scheduler.timesteps:
                # 1. predict noise model_output
                noise_output = trained_diffusion_model(pre_image, t).sample

                # 2. predict previous mean of image x_t-1 and add variance depending on eta
                # eta corresponds to η in paper and should be between [0, 1]
                # do x_t -> x_t-1
                pre_output = noise_scheduler.step(
                    noise_output, t, pre_image
                )
                pre_image = pre_output.prev_sample

        # Denormalize the image after diffusion, from [-1, 1] to [0, 1]
        denormalize = transforms.Normalize([-1, -1, -1], [2, 2, 2])
        denormalized_image = denormalize(pre_image)

        # reshape from [1,3,256,256] to [1,3,H,W]
        denormalized_image = F.interpolate(denormalized_image, size=output_shape, mode='bilinear', align_corners=False)
        # Convert back from [1, C, H, W] to [H, W, C]
        rgb_output = torch.moveaxis(denormalized_image[0].to("cuda"), 0, -1)
        return rgb_output

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        # Convert to float32 and normalize it
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, gt_rgb) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        # Calls a method on the camera optimizer to add additional metrics to the metrics_dict
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
    def get_sdf_loss(self, camera: Cameras, outputs):
        # SDF loss
        self.sdfloss = GS_SDF_Loss(camera,self.gauss_params,outputs,self.radii,self.device).get_gs_sdf_loss()
        return None
    def get_loss_dict(self, outputs, gt_img, metrics_dict=None, use_new_pose = False) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """

        # It is a little counterintuitive that the output is used as gt image to train 3dgs
        # while gs output is the direct output and are used to compute loss
        pred_img = outputs["rgb"]

        # L1 Loss
        Ll1 = torch.abs(gt_img - pred_img).mean()

        # Structural Similarity Index (SSIM) loss
        # SSIM is a perceptual metric that quantifies image quality degradation caused by processing such as data compression or transmission losses. 
        # Instead of absolute pixel differences, SSIM considers changes in structural information, luminance, and contrast.
        
        # permute(2, 0, 1) changes the order of dimensions of the image tensor from (H, W, C) to (C, H, W). i.e.(Channels, Height, Width).
        
        # SSIM values range from -1 to 1, where 1 indicates perfect similarity.
        # By subtracting the SSIM value from 1, the loss increases as similarity decreases.
        # If SSIM is 1 (perfect similarity), the loss is 0. If SSIM is 0, the loss is 1.
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        psnrloss = 1-self.psnr(pred_img, gt_img)

        # with open('/root/losslogs/sdfloss_log.txt', 'a') as f:  # 'a' to append, 'w' to overwrite
        #     f.write(f"{self.sdfloss} \n")
        # with open('/root/losslogs/L1_log.txt', 'a') as f:  # 'a' to append, 'w' to overwrite
        #     f.write(f"{Ll1} \n")
        # with open('/root/losslogs/simloss_log.txt', 'a') as f:  # 'a' to append, 'w' to overwrite
        #     f.write(f"{simloss} \n")
        # with open('/root/losslogs/psnr_log.txt', 'a') as f:  # 'a' to append, 'w' to overwrite
        #     f.write(f"{psnrloss} \n")
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            # Regularization Calculation
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        
        if use_new_pose == True:
            total_w = 1 + self.config.ssim_lambda 
            weight = [1/total_w, self.config.ssim_lambda/total_w]
            L_diff = weight[0] * Ll1  + weight[1] * simloss 
            loss_dict = {
                "main_loss": self.config.diffusion_lambda *L_diff,
                "scale_reg": scale_reg + self.config.sdf_lambda* self.sdfloss
            }
        else:
            total_w = 1 + self.config.ssim_lambda + self.config.psnr_lambda 
            weight = [1/total_w, self.config.ssim_lambda/total_w, self.config.psnr_lambda/total_w]
            loss_dict = {
                "main_loss": weight[0] * Ll1 + weight[1] * simloss + weight[2] * psnrloss,
                "scale_reg": scale_reg + self.config.sdf_lambda* self.sdfloss
            }
        
        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], gt_rgb: torch.Tensor
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        metrics_dict["sdf"] = float(self.sdfloss)
        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
