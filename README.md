# MSc Robotics and Compuatation Dissertation Research Diary

Rules: 
* 10 mins is more than enough 
* Set goals in morning, report at end of day

### Weekwise Diary
- [Week 1](Weekwise%20Diary/Week1.md)
- [Week 2](Weekwise%20Diary/Week2.md)
- [Week 3](Weekwise%20Diary/Week3.md)
- [Week 4](Weekwise%20Diary/Week4.md)
- [Week 5](Weekwise%20Diary/Week5.md)
- [Week 6](Weekwise%20Diary/Week6.md)
- [Week 7](Weekwise%20Diary/Week7.md)
- [Week 8](Weekwise%20Diary/Week8.md)
- [Week 9](Weekwise%20Diary/Week9.md)
- [Week 10](Weekwise%20Diary/Week10.md)
- [Week 11](Weekwise%20Diary/Week11.md)
- [Week 12](Weekwise%20Diary/Week12.md)
- [Week 13](Weekwise%20Diary/Week13.md)

#### Environment: 
Python = 3.10

#### Dependency:
Install PyTorch with CUDA (this repo has been tested with CUDA 11.8) and tiny-cuda-nn. cuda-toolkit is required for building tiny-cuda-nn.
```ruby
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Pytorch3d is required for SDF regularization.(0.7.7 is used for this repo)
```ruby
pip install pytorch3d
```

#### Install [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [gsplat](https://github.com/nerfstudio-project/gsplat) for 3D Gaussian Splatting

#### Code config
After installing Nerfstudio and gsplat, put ddim_splat_sdf folder files in right places in /Nerfstudio folder:

- replace /nerfstudio/nerfstudio/data/dataparsers/blender_dataparser.py with ddim_splat_sdf/blender_dataparser.py 
- put ddim_splat_sdf/ddim_splat_trainer.py in /nerfstudio/nerfstudio/engine/
- put ddim_splat_sdf/gs_sdf_loss.py in /nerfstudio/nerfstudio/model_components/
- put ddim_splat_sdf/ddim_splatfacto.py in /nerfstudio/nerfstudio/models/
- put ddim_splat_sdf/ddim_splat_pipeline.py in /nerfstudio/nerfstudio/pipelines/
- replace nerfstudio/nerfstudio/configs/method_configs.py with method_configs.py

### [Final Dissertation](High_fidelity_Endoscopic_Image_Synthesis_via_3D_Gaussian_Splatting_and_Diffusion_Implicit_Model_Integration.pdf)
