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

pip uninstall torch torchvision functorch tinycudann

# Torch 2.1.2 with CUDA 11.8 
# Install PyTorch 2.1.2 with CUDA 11.8:
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install nvidia/label/cuda-11.8.0::cuda-toolkit
# Install tiny-cuda-nn/gsplat
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

pip install git+https://github.com/ingra14m/depth-diff-gaussian-rasterization.git@depth
pip install git+https://github.com/facebookresearch/pytorch3d.git
