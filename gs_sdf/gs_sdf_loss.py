import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms.rotation_conversions as t3d_rot
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from nerfstudio.models.ddim_splatfacto import k_nearest_sklearn
from nerfstudio.cameras.cameras import Cameras

class GS_SDF_Loss(nn):
    def __init__(self, camera: Cameras, viewmat, gaussians, outputs, radii, device):
        super(GS_SDF_Loss, self).__init__()
        self.camera = camera
        R = viewmat[..., :3, :3]
        T = viewmat[..., :3, 3]
        K = camera.get_intrinsics_matrices()
        self.p3d_cameras = P3DCameras(R=R, T=T, K=K, znear=0.0001)
        # viewmat: world2camera matrix
        self.viewmat = viewmat
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.gaussians = gaussians
        self.outputs = outputs
        self.radii = radii
        self.device = device
        self.squared_sdf_estimation_loss=False
        self.n_samples_for_sdf_regularization = 1_000_000
        self.sdf_sampling_scale_factor = 1.5
    def gs_sdf_loss(self):
        """ 
        get the sdf loss for the given view matrix and gaussians

        self.gaussians = torch.nn.ParameterDict(
        {
            "means": means,
            "scales": scales,
            "quats": quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
        }
        )
        """
        # get the sampling mask
        sampling_mask, gaussian_standard_deviations = self.get_sampling_mask(self.outputs["depth"], self.gaussians["means"], self.gaussians["scales"], self.gaussians["quats"])
        # get the sdf samples
        sdf_samples, sdf_gaussian_idx = self.sample_points_in_gaussians(
            num_samples=self.n_samples_for_sdf_regularization, 
            sampling_scale_factor=self.sdf_sampling_scale_factor,
            mask=sampling_mask,
            probabilities_proportional_to_volume=False,
            )
        # get the sdf values
        sdf_values= self.get_sdf_values(self.gaussians["means"], self.gaussians["scales"], self.gaussians["quats"], sdf_samples, sdf_sample_idx)
        # get the sdf estimation
        sdf_estimation, proj_mask = self.get_sdf_estimation(sdf_samples)
        # normalize the sdf values by the sdf sample standard deviation
        sdf_sample_std = gaussian_standard_deviations[sdf_gaussian_idx][proj_mask]
        if self.squared_sdf_estimation_loss:
            sdf_estimation_loss = ((sdf_values - sdf_estimation.abs()) / sdf_sample_std).pow(2)
        else:
            sdf_estimation_loss = (sdf_values - sdf_estimation.abs()).abs() / sdf_sample_std
        return sdf_estimation_loss.clamp(max=10.).mean()
    def get_sdf_values(self, means, scaling, quats, sdf_samples, gaussian_idx, density_factor=1./16., density_threshold=1, opacity_min_clamp=1e-16):
        """ 
        get the sdf values for the given view matrix

        Args:
        sdf_samples: sdf samples
        gaussian_idx: sdf sample index
        """
        # get the closest gaussians
        gaussian_strengths = 0.1 * torch.ones_like(means[:,0]).view(-1, 1)
        gaussian_centers = means
        gaussian_inv_scaled_rotation = self.get_covariance(quats, scaling, return_full_matrix=True, return_sqrt=True, inverse_scales=True)
        
        _, closest_gaussians_idx = k_nearest_sklearn(gaussian_centers, 16)[gaussian_idx]
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx]
        closest_gaussian_strengths = gaussian_strengths[closest_gaussians_idx]
            
        # return closest_gaussian_opacities:
        shift = (sdf_samples[:, None] - closest_gaussian_centers)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        density_mask = densities >= 1.
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)
        
        beta = scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
        clamped_densities = densities.clamp(min=opacity_min_clamp)

        # compute the sdf values
        sdf_values = beta * (
                        torch.sqrt(-2. * torch.log(clamped_densities)) # TODO: Change the max=1. to something else?
                        - np.sqrt(-2. * np.log(min(density_threshold, 1.)))
                        )
        return sdf_values

    def get_sdf_estimation(self,sdf_samples):
        # Compute the depth of the points in the gaussians
        sdf_samples_in_camera_space = self.p3d_cameras.get_world_to_view_transform().transform_points(sdf_samples)
        sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.
        proj_mask = sdf_samples_z > self.p3d_cameras.znear
        sdf_samples_map_z = self.get_points_depth_in_depth_map(self.outputs["depth"], sdf_samples_in_camera_space[proj_mask])
        sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]
        return sdf_estimation, proj_mask
    
    def get_covariance(self, quats, scaling, return_full_matrix=False, return_sqrt=False, inverse_scales=False):
        if inverse_scales:
            scaling = 1. / scaling.clamp(min=1e-8)
        scaled_rotation = t3d_rot.quaternion_to_matrix(quats) * scaling[:, None]
        if return_sqrt:
            return scaled_rotation
        
        cov3Dmatrix = scaled_rotation @ scaled_rotation.transpose(-1, -2)
        if return_full_matrix:
            return cov3Dmatrix
        
        cov3D = torch.zeros((cov3Dmatrix.shape[0], 6), dtype=torch.float, device=self.device)
        cov3D[:, 0] = cov3Dmatrix[:, 0, 0]
        cov3D[:, 1] = cov3Dmatrix[:, 0, 1]
        cov3D[:, 2] = cov3Dmatrix[:, 0, 2]
        cov3D[:, 3] = cov3Dmatrix[:, 1, 1]
        cov3D[:, 4] = cov3Dmatrix[:, 1, 2]
        cov3D[:, 5] = cov3Dmatrix[:, 2, 2]
        
        return cov3D

    def get_points_depth_in_depth_map(self, depth, points_in_camera_space):
            """Projecting the 3D points onto the 2D image plane using the camera's projection matrix.
                Normalizing these 2D projections to fit the format required by grid_sample.
                Using grid_sample to retrieve the depth values from the depth map at the projected locations."""
            depth_view = depth.unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2)
            pts_projections = self.p3d_cameras.get_projection_transform().transform_points(points_in_camera_space)

            factor = -1 * min(self.image_height, self.image_width)
            # todo: Parallelize these two lines with a tensor [image_width, image_height]
            pts_projections[..., 0] = factor / self.image_width * pts_projections[..., 0]
            pts_projections[..., 1] = factor / self.image_height * pts_projections[..., 1]
            pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

            map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                    grid=pts_projections,
                                                    mode='bilinear',
                                                    padding_mode='border'  # 'reflection', 'zeros'
                                                    )[0, 0, :, 0]
            return map_z
        
    def sample_points_in_gaussians(self, num_samples, sampling_scale_factor=1., mask=None,
                                    probabilities_proportional_to_opacity=False,
                                    probabilities_proportional_to_volume=True,):
        """Sample points in the Gaussians.

        Args:
            num_samples (_type_): _description_
            sampling_scale_factor (_type_, optional): _description_. Defaults to 1..
            mask (_type_, optional): _description_. Defaults to None.
            probabilities_proportional_to_opacity (bool, optional): _description_. Defaults to False.
            probabilities_proportional_to_volume (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        if mask is None:
            scaling = self.scaling
        else:
            scaling = self.scaling[mask]
        
        if probabilities_proportional_to_volume:
            areas = scaling[..., 0] * scaling[..., 1] * scaling[..., 2]
        else:
            areas = torch.ones_like(scaling[..., 0])
        
        if probabilities_proportional_to_opacity:
            if mask is None:
                areas = areas * self.strengths.view(-1)
            else:
                areas = areas * self.strengths[mask].view(-1)
        areas = areas.abs()
        # cum_probs = areas.cumsum(dim=-1) / areas.sum(dim=-1, keepdim=True)
        cum_probs = areas / areas.sum(dim=-1, keepdim=True)
        
        random_indices = torch.multinomial(cum_probs, num_samples=num_samples, replacement=True)
        if mask is not None:
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]
        
        random_points = self.points[random_indices] + t3d_rot.quaternion_apply(
            self.quaternions[random_indices], 
            sampling_scale_factor * self.scaling[random_indices] * torch.randn_like(self.points[random_indices]))
        
        return random_points, random_indices
    
    def get_sampling_mask(self, depth, points, scaling, quaternions, sample_only_in_gaussians_close_to_surface=True, close_gaussian_threshold=2.):
        visibility_mask = self.radii > 0
        if sample_only_in_gaussians_close_to_surface:
            with torch.no_grad():
                gaussian_to_camera = torch.nn.functional.normalize(self.p3d_cameras.get_camera_center() - points, dim=-1)
                gaussian_centers_in_camera_space = self.p3d_cameras.get_world_to_view_transform().transform_points(points)
                
                gaussian_centers_z = gaussian_centers_in_camera_space[..., 2] + 0.
                gaussian_centers_map_z = self.get_points_depth_in_depth_map(depth, gaussian_centers_in_camera_space)
                
                gaussian_standard_deviations = (
                    scaling * t3d_rot.quaternion_apply(t3d_rot.quaternion_invert(quaternions), gaussian_to_camera)
                    ).norm(dim=-1)
            
                gaussians_close_to_surface = (gaussian_centers_map_z - gaussian_centers_z).abs() < close_gaussian_threshold * gaussian_standard_deviations
                sampling_mask = sampling_mask * gaussians_close_to_surface
        else:
            sampling_mask = visibility_mask
        return sampling_mask, gaussian_standard_deviations