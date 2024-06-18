
----------
###### Title: 2024 Robotics and Computation Dissertation - Week 6
###### Date: 17-06-2024 -- 21-06-2024
----------
###### Monday
#### Nerfstudio configuration on computer from start

##### Create environment
```ruby
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```
##### Dependencies
```ruby
pip uninstall torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

##### Install tiny-cuda-nn/gsplat

Download
```ruby
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
```
Problems occured: Cuda version installed in the conda environment is not used, instead the cuda in /usr/local/ is used.

Usefull commands
```ruby
sudo update-alternatives --display cuda  # display alternatives
sudo update-alternatives --config cuda
nvcc --version  # check present cuda version
nano ~/.bashrc  # open .bashrc file and modify
```

Comment out the following in .bashrc file
```ruby
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Add the following to .bashrc file
```ruby
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
```
Then, apply changes with ```source ~/.bashrc```

Another error:
```ruby
(nerfstudio) wangzican@DESKTOP-0QF7VTC:~/tiny-cuda-nn$ cd bindings/torch
python setup.py install
setup.py:5: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
  from pkg_resources import parse_version
Traceback (most recent call last):
  File "setup.py", line 11, in <module>
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
  File "/home/wangzican/miniconda3/envs/nerfstudio/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 28, in <module>
    from pkg_resources import packaging  # type: ignore[attr-defined]
ImportError: cannot import name 'packaging' from 'pkg_resources' (/home/wangzican/miniconda3/envs/nerfstudio/lib/python3.8/site-packages/pkg_resources/__init__.py)
(nerfstudio) wangzican@DESKTOP-0QF7VTC:~/tiny-cuda-nn/bindings/torch$ pip install setuptools==66.1.1  # Solved by reinstalling the version which don't have these deprecated warnings.
```

Still, a little problem
```ruby 
error: [Errno 2] No such file or directory: '/home/wangzican/miniconda3/bin/nvcc
```

Solved by changing
```ruby
export CUDA_HOME=$CONDA_PREFIX
```
to

```ruby
export CUDA_HOME=/home/wangzican/miniconda3/envs/nerfstudio
```

Decided to change to cuda 11.7

Still couldn't run setup.py in tiny-cuda-nn:
```ruby
conda install -c conda-forge setuptools wheel numpy
pip install --upgrade setuptools packaging
# Replace pkg_resources with packaging: Modify the import statement to use packaging directly.
# from pkg_resources import parse_version
from packaging.version import parse as parse_version
```

Another error:
```ruby
cannot find -lcuda: No such file or directory
collect2: error: ld returned 1 exit status
error: command '/usr/bin/g++' failed with exit code 1
```

```ruby
# Find libcuda.so file
sudo find / -name "libcuda.so*" 2>/dev/null

/usr/lib/wsl/lib/libcuda.so
/usr/lib/wsl/lib/libcuda.so.1
/usr/lib/wsl/lib/libcuda.so.1.1
/usr/lib/wsl/drivers/nvdd.inf_amd64_67b1df330bec74ef/libcuda.so.1.1
/usr/local/cuda-12.1/targets/x86_64-linux/lib/stubs/libcuda.so
/usr/local/cuda-12.5/targets/x86_64-linux/lib/stubs/libcuda.so
/home/wangzican/miniconda3/pkgs/cuda-driver-dev_linux-64-12.4.127-hd681fbe_0/targets/x86_64-linux/lib/stubs/libcuda.so
/home/wangzican/miniconda3/pkgs/cuda-driver-dev-11.8.89-0/lib/stubs/libcuda.so
/home/wangzican/miniconda3/pkgs/cuda-driver-dev-12.4.127-h99ab3db_0/lib/stubs/libcuda.so
/home/wangzican/miniconda3/envs/nerfstudio/pkgs/cuda-toolkit/targets/x86_64-linux/lib/stubs/libcuda.so
/home/wangzican/miniconda3/envs/nerfstudio/lib/stubs/libcuda.so
/home/wangzican/miniconda3/envs/nerfstudio/targets/x86_64-linux/lib/stubs/libcuda.so

# Create symlinks in /usr/lib
sudo ln -sf /usr/lib/wsl/lib/libcuda.so /usr/lib/libcuda.so
sudo ln -sf /usr/lib/wsl/lib/libcuda.so.1 /usr/lib/libcuda.so.1
```

Successfully processing dependencies for tinycudann==1.7 ```pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch```

##### Installing nerfstudio
From pip
```ruby
pip install nerfstudio
```

From source Optional, use this command if you want the latest development version.
```ruby
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```
#### Tab completion (bash & zsh)

This needs to be rerun when the CLI changes, for example if nerfstudio is updated.

```ns-install-cli```
##### Development packages
```ruby
pip install -e .[dev]
pip install -e .[docs]
```

##### Download Colmap and FFmpeg
```ruby

# Installation of colmap
conda install -c conda-forge colmap
colmap -h # To check

# Installation of FFmpeg
sudo apt update && sudo apt upgrade
sudo apt install ffmpeg
ffmpeg -version
```
#### Run nerf in nerfstudio
```ruby
# Process data
ns-process-data images --data /home/wangzican/data/turtles/raw_images --output-dir /home/wangzican/data/turtles
# train data
ns-train nerfacto --data /home/wangzican/data/turtles
```

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week8.md)
