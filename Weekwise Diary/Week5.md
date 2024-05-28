
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
  
  ![image](https://github.com/QinyanGong/MScRoboticsandComputationDissertation/assets/74662060/790eb47e-7a5a-4d51-b13d-93d6ac6030bf)

  Finish 4-epoch training, but still have problems on model saving.

&nbsp;
----------
###### Tuesday
- Modify ["train.py"](../train.py): previous code have failed to callback, so impossible to save the checkpoints.
![image](https://github.com/QinyanGong/MScRoboticsandComputationDissertation/assets/74662060/99055d18-ef23-4481-bb5f-a3165b845c90)
- Run the testing steps and generate a .gif file and a sequence of images in nerf_pl/results/llff/$SCENE
 <img src="C3VD testing output 00.png" alt="C3VD testing output 00" width="500" height="325">
 <img src="C3VD.gif" width="250" height="250"/>
  The output is very weird, perhaps because the dataset used for training is too small.
&nbsp;
----------
###### Thursday
&nbsp;
----------
###### Friday
&nbsp;
----------
&nbsp;
> ###### [Next Week](Week6.md)
