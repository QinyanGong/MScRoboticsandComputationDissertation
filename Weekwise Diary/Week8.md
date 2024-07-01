remove environment:

```ruby
# Deactivate the environment if it is active
conda deactivate

# Remove the environment
conda env remove --name myenv


# Clear Conda Cache and Reinstall:
# Clear the Conda cache and reinstall PyTorch. This ensures any corrupted or incomplete packages are removed:
conda clean --all
conda uninstall pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia


# Verify that the environment has been removed
conda env list
```
Config the EndoGS environment the same process as nerfstudio
```ruby
pip uninstall torch torchvision functorch tinycudann

# Torch 2.1.2 with CUDA 11.8 
# Install PyTorch 2.1.2 with CUDA 11.8:
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install cuda toolkit from [conda website](https://anaconda.org/nvidia/cuda-toolkit)
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

# Use different cuda
alias use_endogs_nvcc='export CUDA_HOME=/home/wangzican/miniconda3/envs/EndoGS'
alias use_nerfstudio_nvcc='export CUDA_HOME=/home/wangzican/miniconda3/envs/nerfstudio'
# Now, you can easily switch between the nvcc binaries by running:
use_endogs_nvcc

# Install tiny-cuda-nn/gsplat
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
# or install from [conda](https://github.com/conda-forge/ninja-feedstock)
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install ninja


pip install git+https://github.com/ingra14m/depth-diff-gaussian-rasterization.git@depth
```

Install pytorch3d following the instruction on github [website](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```ruby
conda install pytorch3d -c pytorch3d
```

ERROR: fail to identify torchvision
Find the matching versions of torch and cuda on https://pytorch.org/get-started/previous-versions/
```ruby
# Uninstall torch and install with code on official website
conda uninstall pytorch torchvision torchaudio
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```
 ```ruby
ImportError: cannot import name 'is_arabic' from 'charset_normalizer.utils' (/home/wangzican/miniconda3/envs/EndoGS/lib/python3.8/site-packages/charset_normalizer/utils.py)
 pip install chardet
```

#### Fine Review of 3D Gaussian Splatting

- [Ray Marching](https://michaelwalczyk.com/blog-ray-marching.html)
- [What do we mean by isotropic/anisotropic covariance?](https://statisticaloddsandends.wordpress.com/2019/10/23/what-do-we-mean-by-isotropic-anisotropic-covariance/)
- Gaussians are defined by a full 3D covariance matrix Σ defined in world space centered at point (mean): [EWA Volume Splatting](https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf)
- [Plenoxels: Radiance Fields without Neural Networks Alex](https://arxiv.org/abs/2112.05131)
