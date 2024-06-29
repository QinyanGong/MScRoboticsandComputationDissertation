remove environment:

```ruby
# Deactivate the environment if it is active
conda deactivate

# Remove the environment
conda env remove --name myenv

# Verify that the environment has been removed
conda env list
```
Config the EndoGS environment the same process as nerfstudio
```ruby
pip uninstall torch torchvision functorch tinycudann

# Torch 2.1.2 with CUDA 11.8 
# Install PyTorch 2.1.2 with CUDA 11.8:
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install nvidia/label/cuda-11.8.0::cuda-toolkit
# Install tiny-cuda-nn/gsplat
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

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
