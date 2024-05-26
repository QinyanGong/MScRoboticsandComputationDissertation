
----------
###### Title: 2024 Robotics and Computation Dissertation - Week 5
###### Date: 26-05-2024 -- 31-05-2024
----------
###### Monday
- Run Colmap and [Zoomlab Code](https://github.com/superrice2020/ZoomLab_NeRF) to get poses_bounds.npy for nerf training

- Modify AIKui's [nerf code](https://github.com/kwea123/nerf_pl) on Github to train C3VD:
  1. clone the repository
  2. change the ["requirements"](../requirements.txt)
  3. change the ["train.py"](../train.py)
  4. run the codes

  <img src="aikui nerf C3VD 0-6 Epoch 256 batchsize 4096 data.png" alt="aikui nerf C3VD 0-6 Epoch 256 batchsize 4096 data" width="300" height="325">

  <img src="aikui nerf C3VD 6-9 Epoch 256 batchsize 4096 data.png" alt="aikui nerf C3VD 6-9 Epoch 256 batchsize 4096 data" width="500" height="325">

  Finish 10-epoch training, but run out of capacity when post-processing.
  Still have problems on model saving.

&nbsp;
----------
###### Thursday

----------
###### Friday
----------
&nbsp;
> ###### [Next Week](Week6.md)
