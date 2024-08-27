----------
###### Title: 2024 Robotics and Computation Dissertation - Week 12
###### Date: 25-08-2024 -- 30-08-2024
----------
###### Monday-Sunday

##### Modify the code to make the diffusion loss act as a prior regularising the update

- Modify get_loss_dict related code in ddim_splatfacto and ddim_splat_pipeline
- Create new trainer and train file: ddim_splat_trainer, to use diffusion gradients in normal update as regularizer.
- Change the TrainerConfig of ddim_splat model in methodconfig file to DDIMSplat_TrainerConfig
- ddim_splat_trainer is the trainer class that is modified train_iteration function

##### Experiments and Results
- result00: /home/wangzican/outputs/cecum_t1_a/ddim_splatfacto/2024-08-25_225314
- config: diff_lambda = 0.01; gradient discount:0.05; p_novel_camera:0.1

- result01:/home/wangzican/outputs/cecum_t1_a/ddim_splatfacto/2024-08-26_000832
- config: diff_lambda = 1e-6; gradient discount:0.05; p_novel_camera:0.01

##### Further Modification: 
1. do not optimize if use novel-view images
2. sample noise from common source:
    - generate 3D noise
    - rasterize to 2D noise
    - add to rendered image

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week14.md)
