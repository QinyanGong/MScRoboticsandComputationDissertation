from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Type, Any, Dict,List, Tuple, Union

from pathlib import Path
from time import time

import torch
import torchvision.utils as vutils
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes
from nerfstudio.utils import profiler
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.ddim_splatfacto import DDIMSplatfactoModel

@dataclass
class DDIMSplatPipelineConfig(VanillaPipelineConfig):
    """DDIM Splatfacto Pipeline Config"""

    _target: Type = field(default_factory=lambda: DDIMSplatPipeline)
    p_novel_camera: float = 0.1 #0.1, 0.2, 0.01
    """it has 0.1 probability to train with novel camera pose images"""


class DDIMSplatPipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""
    config: DDIMSplatPipelineConfig
    model: DDIMSplatfactoModel
    datamanager: FullImageDatamanager
    def __init__(
        self,
        config: DDIMSplatPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        self.use_new_pose = True


    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError


    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        if self.world_size > 1 and step:
            assert self.datamanager.train_sampler is not None
            self.datamanager.train_sampler.set_epoch(step)
        camera, batch = self.datamanager.next_train(step)
        camera = self.perturb_cameras(camera)
        model_outputs = self.model(camera)

        # if we use new pose, the ground truth image is the diffusion image
        # else, we use the original image
        if self.use_new_pose == True:
            df_outputs = self.model.get_df_outputs(camera)
            gt_rgb = df_outputs["rgb"]
        else:
            gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), model_outputs["background"])
        self.model.get_sdf_loss(camera, model_outputs)
        metrics_dict = self.model.get_metrics_dict(model_outputs, gt_rgb)
        loss_dict = self.model.get_loss_dict(model_outputs, gt_rgb, metrics_dict, self.use_new_pose)

        return model_outputs, loss_dict, metrics_dict
    @profiler.time_function
    def get_eval_loss_dict(self, step: int) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval(step)

        # output directly from 3DGS to evaluate:
        model_outputs = self.model(camera)

        # we use the original image for evaluation
        gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), model_outputs["background"])

        metrics_dict = self.model.get_metrics_dict(model_outputs, gt_rgb)
        loss_dict = self.model.get_loss_dict(model_outputs, gt_rgb, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict


    @profiler.time_function
    def get_eval_image_metrics_and_images(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model(camera)

        # we use the original image for evaluation
        gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), outputs["background"])

        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, gt_rgb)
        assert "num_rays" not in metrics_dict
        metrics_dict["num_rays"] = (camera.height * camera.width * camera.size).item()
        self.train()
        return metrics_dict, images_dict

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                # camera = self.perturb_cameras(camera)
                outputs = self.model(camera)
                
                # # if we use new pose, the ground truth image is the diffusion image
                # # else, we use the original image
                # if self.use_new_pose == True:
                #     df_outputs = self.model.get_df_outputs(camera)
                #     gt_rgb = df_outputs["rgb"]
                # else:
                gt_rgb = self.model.composite_with_background(self.model.get_gt_img(batch["image"]), outputs["background"])
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, gt_rgb)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.train()
        return metrics_dict


    def perturb_cameras(self, cameras: Cameras):
        # To train model like dyna in RL
        # make it have certain probability to choose original camera data and novel camera poses
        p_original = torch.rand(1)
        if(p_original > self.config.p_novel_camera):
            # use original camera poses
            self.use_new_pose = False
            return cameras
        else:
            # Ensure that the new random tensor is on the same device as cameras.camera_to_worlds
            random_tensor = (torch.rand(cameras.camera_to_worlds.shape) - 0.5) * 0.05
            random_tensor = random_tensor.to(cameras.camera_to_worlds.device)

            # Perform the addition
            cameras.camera_to_worlds = random_tensor + cameras.camera_to_worlds    
            
            self.use_new_pose = True        

        return cameras