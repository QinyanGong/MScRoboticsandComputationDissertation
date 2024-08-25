----------
###### Title: 2024 Robotics and Computation Dissertation - Week 12
###### Date: 12-08-2024 -- 16-08-2024
----------
###### Monday-Sunday

##### Modify the code to make the diffusion loss act as a prior regularising the update

- Modify get_loss_dict related code in ddim_splatfacto and ddim_splat_pipeline
- Create new trainer and train file: ddim_splat_trainer, to use diffusion gradients in normal update as regularizer.
- Change the TrainerConfig of ddim_splat model in methodconfig file to DDIMSplat_TrainerConfig
- ddim_splat_trainer is the trainer class that is modified train_iteration function

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week13.md)
