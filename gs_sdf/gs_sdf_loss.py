import math
import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms.rotation_conversions as t3d_rot
from pytorch3d.renderer import FoVPerspectiveCameras as P3DCameras
from nerfstudio.cameras.cameras import Cameras
from pytorch3d.renderer.cameras import _get_sfm_calibration_matrix
from pytorch3d.ops import knn_points

class GS_SDF_Loss(nn.Module):
    def __init__(self, camera: Cameras, gaussians, outputs, radii, device):
        super().__init__()
        self.p3d_cameras = self.convert_camera_from_gs_to_pytorch3d(camera,device=device)
        self.image_height = camera.image_height
        self.image_width = camera.image_width
        self.gaussians = gaussians
        self.outputs = outputs
        self.depth = outputs["depth"]
        self.radii = radii.to(device)
        self.device = device
        self.squared_sdf_estimation_loss=False
        self.n_samples_for_sdf_regularization = 1_000_000
        self.sdf_sampling_scale_factor = 1.5
        self.scaling = gaussians.scales.detach()
        self.points = gaussians.means.detach()
        self.quaternions = gaussians.quats.detach()
        self.n_points = gaussians.means.shape[0]

    
    def fov2focal(self, fov, pixels):
        return pixels / (2 * math.tan(fov / 2))
    def convert_camera_from_gs_to_pytorch3d(self, gs_camera: Cameras, device='cuda'):
        """
        From Gaussian Splatting camera parameters,
        computes R, T, K matrices and outputs pytorch3d-compatible camera object.

        Args:
            gs_cameras (List of GSCamera): List of Gaussian Splatting cameras.
            device (_type_, optional): _description_. Defaults to 'cuda'.

        Returns:
            p3d_cameras: pytorch3d-compatible camera object.
        """
        
        N = len(gs_camera)
        
        R = gs_camera.camera_to_worlds[:, :3, :3].to(device)
        T = gs_camera.camera_to_worlds[:, :3, 3].to(device)
        R_inv = R.transpose(1, 2)
        # print("R shape:",R.shape)
        # print("T shape:",T.unsqueeze(2).shape)
        T_inv = -torch.bmm(R_inv, T.unsqueeze(2))
        T_inv = T_inv.squeeze(2)
        fx = gs_camera.fx.to(device)
        fy = gs_camera.fy.to(device)
        image_height = gs_camera.image_height.to(device)
        image_width = gs_camera.image_width.to(device)
        cx = gs_camera.cx  # torch.zeros_like(fx).to(device)
        cy = gs_camera.cy  # torch.zeros_like(fy).to(device)
        
        # c2w = gs_camera.camera_to_worlds.to(device)
        
        # distortion_params = torch.zeros(N, 6).to(device)
        # camera_type = torch.ones(N, 1, dtype=torch.int32).to(device)

        # Pytorch3d-compatible camera matrices
        # Intrinsics
        image_size = torch.Tensor(
            [image_width[0], image_height[0]],
        )[
            None
        ].to(device)
        scale = image_size.min(dim=1, keepdim=True)[0] / 2.0
        c0 = image_size / 2.0
        p0_pytorch3d = (
            -(
                torch.Tensor(
                    (cx[0], cy[0]),
                )[
                    None
                ].to(device)
                - c0
            )
            / scale
        )
        focal_pytorch3d = (
            torch.Tensor([fx[0], fy[0]])[None].to(device) / scale
        )
        K = _get_sfm_calibration_matrix(
            1, "cuda", focal_pytorch3d, p0_pytorch3d, orthographic=False
        )
        K = K.expand(N, -1, -1)

        p3d_cameras = P3DCameras(device=device, R=R_inv, T=T_inv, K=K, znear=0.0001)

        return p3d_cameras
    def get_gs_sdf_loss(self):
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
        sampling_mask, gaussian_standard_deviations = self.get_sampling_mask(self.depth, self.points, self.scaling, self.quaternions)
        # get the sdf samples
        sdf_samples, sdf_gaussian_idx = self.sample_points_in_gaussians(
            num_samples=self.n_samples_for_sdf_regularization, 
            sampling_scale_factor=self.sdf_sampling_scale_factor,
            mask=sampling_mask,
            probabilities_proportional_to_volume=False,
            )
        # get the sdf values
        sdf_values= self.get_sdf_values(self.gaussians["means"], self.gaussians["scales"], self.gaussians["quats"], sdf_samples, sdf_gaussian_idx)
        # print("sdf_values:", sdf_values)
        # get the sdf estimation
        sdf_estimation, proj_mask = self.get_sdf_estimation(sdf_samples)
        # print("sdf_values:", sdf_estimation)
        # normalize the sdf values by the sdf sample standard deviation
        sdf_sample_std = gaussian_standard_deviations[sdf_gaussian_idx][proj_mask]
        if self.squared_sdf_estimation_loss:
            sdf_estimation_loss = ((sdf_estimation.abs() - sdf_values[proj_mask]) / sdf_sample_std).pow(2)
        else:
            sdf_estimation_loss = (sdf_estimation.abs() - sdf_values[proj_mask]).abs() / sdf_sample_std
        return sdf_estimation_loss.clamp(max=10.).mean()
    def get_sdf_values(self, gaussian_centers, scaling, quats, sdf_samples, gaussian_idx, density_factor=1./16., density_threshold=1, opacity_min_clamp=1e-16):
        """ 
        get the sdf values for the given view matrix

        Args:
        sdf_samples: sdf samples
        gaussian_idx: sdf sample index
        """

        # get the closest gaussians
        gaussian_strengths = 0.1 * torch.ones_like(gaussian_centers[:,0]).view(-1, 1)
        gaussian_inv_scaled_rotation = self.get_covariance(quats, scaling, return_full_matrix=True, return_sqrt=True, inverse_scales=True)
        
        with torch.no_grad():
            knns = knn_points(self.points[None], self.points[None], K=16)
            closest_gaussians_idx = knns.idx[0]
        closest_gaussians_idx = closest_gaussians_idx[gaussian_idx]
        closest_gaussian_centers = gaussian_centers[closest_gaussians_idx]
        closest_gaussian_inv_scaled_rotation = gaussian_inv_scaled_rotation[closest_gaussians_idx]
        closest_gaussian_strengths = gaussian_strengths[closest_gaussians_idx]
            
        # return closest_gaussian_opacities:
        # Compute the density field as a sum of local gaussian opacities
        shift = (sdf_samples[:, None] - closest_gaussian_centers)
        warped_shift = closest_gaussian_inv_scaled_rotation.transpose(-1, -2) @ shift[..., None]
        neighbor_opacities = (warped_shift[..., 0] * warped_shift[..., 0]).sum(dim=-1).clamp(min=0., max=1e8)
        neighbor_opacities = density_factor * closest_gaussian_strengths[..., 0] * torch.exp(-1. / 2 * neighbor_opacities)
        densities = neighbor_opacities.sum(dim=-1)
        density_mask = densities >= 1.
        densities[density_mask] = densities[density_mask] / (densities[density_mask].detach() + 1e-12)
        
        # beta on the same device as scaling
        beta = scaling.min(dim=-1)[0][closest_gaussians_idx].mean(dim=1)
        clamped_densities = densities.clamp(min=opacity_min_clamp)
        # print("sdf samples shape",sdf_samples.shape)
        # print("gaussian index shape:",gaussian_idx.shape)
        # print("closest gaussians index shape:", closest_gaussians_idx.shape)
        # print("closest gaussaian centers shape:", closest_gaussian_centers.shape)
        # print("neighbor opacities shape:", neighbor_opacities.shape)
        # print("densities shape:", densities.shape)
        # compute the sdf values
        sdf_values = beta * (
                        torch.sqrt(-2. * torch.log(clamped_densities)) # TODO: Change the max=1. to something else?
                        - torch.sqrt(-2. * torch.log(torch.tensor(min(density_threshold, 1.), device=self.device)))
                        )
        # returns are on self.device
        return sdf_values

    def get_sdf_estimation(self,sdf_samples):
        # Compute the depth of the points in the gaussians
        sdf_samples_in_camera_space = self.p3d_cameras.get_world_to_view_transform().transform_points(sdf_samples)
        sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.
        proj_mask = sdf_samples_z > self.p3d_cameras.znear
        # sdf_samples_map_z is on self.device
        sdf_samples_map_z = self.get_points_depth_in_depth_map(self.depth, sdf_samples_in_camera_space[proj_mask])
        sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]
        # print("sdf estimation shape",sdf_estimation.shape)
        # print("proj mask shape",proj_mask.shape)
        # returns are on self.device
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
        # returns are on self.device
        return cov3D

    def get_points_depth_in_depth_map(self, depth, points_in_camera_space):
            """Projecting the 3D points onto the 2D image plane using the camera's projection matrix.
                Normalizing these 2D projections to fit the format required by grid_sample.
                Using grid_sample to retrieve the depth values from the depth map at the projected locations."""
            # [batch_size, H, W] -> [1,1,H,W]
            depth_view = depth.unsqueeze(0)  
            pts_projections = self.p3d_cameras.get_projection_transform().transform_points(points_in_camera_space)

            factor = -1 * min(self.image_height, self.image_width)
            # todo: Parallelize these two lines with a tensor [image_width, image_height]
            pts_projections[..., 0] = factor / self.image_width * pts_projections[..., 0]
            pts_projections[..., 1] = factor / self.image_height * pts_projections[..., 1]
            pts_projections = pts_projections[..., :2].view(1, -1, 1, 2)

            map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                    grid=pts_projections,
                                                    mode='bilinear',
                                                    padding_mode='border',  # 'reflection', 'zeros'
                                                    align_corners=True
                                                    )[0, 0, :, 0]
            # returns are on self.device
            return map_z.to(self.device)
        
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
        
        random_indices = torch.multinomial(cum_probs.to(self.device), num_samples=num_samples, replacement=True)
        if mask is not None:
            valid_indices = torch.arange(self.n_points, device=self.device)[mask]
            random_indices = valid_indices[random_indices]
        
        random_points = self.points[random_indices] + t3d_rot.quaternion_apply(
            self.quaternions[random_indices], 
            sampling_scale_factor * self.scaling[random_indices] * torch.randn_like(self.points[random_indices])).to(self.device)
        # returns are on self.device
        return random_points, random_indices
    
    def get_sampling_mask(self, depth, points, scaling, quaternions, sample_only_in_gaussians_close_to_surface=True, close_gaussian_threshold=2.):
        visibility_mask = self.radii > 0
        sampling_mask = visibility_mask
        if sample_only_in_gaussians_close_to_surface:
            with torch.no_grad():
                gaussian_to_camera = torch.nn.functional.normalize(self.p3d_cameras.get_camera_center() - points, dim=-1)
                gaussian_centers_in_camera_space = self.p3d_cameras.get_world_to_view_transform().transform_points(points)
                
                gaussian_centers_z = gaussian_centers_in_camera_space[..., 2] + 0.
                # gaussian_centers_map_z is on self.device
                gaussian_centers_map_z = self.get_points_depth_in_depth_map(depth, gaussian_centers_in_camera_space)
                
                gaussian_standard_deviations = (
                    scaling * t3d_rot.quaternion_apply(t3d_rot.quaternion_invert(quaternions), gaussian_to_camera)
                    ).norm(dim=-1)
            
                gaussians_close_to_surface = (gaussian_centers_map_z - gaussian_centers_z.to(self.device)).abs() < close_gaussian_threshold * gaussian_standard_deviations
                sampling_mask = sampling_mask * gaussians_close_to_surface
        # returns are on self.device
        return sampling_mask, gaussian_standard_deviations.to(self.device)
    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        # x.cpu().numpy(): Moves the tensor to the CPU and converts it to a NumPy array. 
        # This is necessary because scikit-learn works with NumPy arrays, not PyTorch tensors.
        x_np = x.cpu().detach().numpy()

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
        # returns are on self.device
        distances = torch.tensor(distances[:, 1:], dtype=torch.float32, device=self.device)
        indices = torch.tensor(indices[:, 1:], dtype=torch.long, device=self.device)
        return distances, indices
