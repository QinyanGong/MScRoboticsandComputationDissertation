----------
###### Title: 2024 Robotics and Computation Dissertation - Week 13
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



export CUDA_HOME=/home/wangzican/miniconda3/envs/nerfstudio
export PATH=$CONDA_PREFIX/bin:/home/wangzican/miniconda3/envs/nerfstudio/lib/stubs:$PATH:/home/wangzican/miniconda3/envs/nerfstudio/lib
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.9/site-packages:/home/wangzican/miniconda3/envs/nerfstudio/lib
export PYTHONPATH=$PYTHONPATH:/home/wangzican/miniconda3/envs/nerfstudio/lib/python3.9:/home/wangzican/miniconda3/envs/nerfstudio/lib/python3.9/lib-dynload:\
/home/wangzican/miniconda3/envs/nerfstudio/lib/python3.9/site-packages

Command to train the ddim_splatfacto model

'''
ns-train ddim_splatfacto --pipeline.model.df-img-output-dir /root/renders/cecum_t1_a/df_gt --max-num-iterations 3000 --experiment-name cecum_t1_a --load-dir /root/outputs/cecum_t1_a/splatfacto/2024-09-07_051834/nerfstudio_models  blender-data --data /root/hopodata/c1_a 

ns-train ddim_splatfacto --max-num-iterations 3000 --experiment-name cecum_t1_a --load-dir /root/outputs/cecum_t1_a/splatfacto/2024-09-07_051834/nerfstudio_models blender-
data --steps-per-save --data /root/hopodata/c1_a 
'''


&nbsp;
----------
&nbsp;
> ###### [Next Week](Week14.md)
