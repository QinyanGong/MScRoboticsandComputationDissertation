----------
###### Title: 2024 Robotics and Computation Dissertation - Week 12
###### Date: 12-08-2024 -- 16-08-2024
----------
###### Monday-Sunday

The training result is not good, and it seems that further training will only make it perform worse. Let's find out the reasons.

##### Experiment with different hyperparameters for diffusion model in the ddim_splat loop

###### First test

- Select images from novel view with 0.4 probability
- timesteps for adding noise: 50
- timesteps for denoise: 12
- Noise level: sample from standard normal distribution * 4
- Results:
  
| Non-diffused Image   | Diffused Image |
| ------------- | ------------ |
|  <img src="non_diffused_3dsplat_image00.png" alt="non_diffused_3dsplat_image00" width="225" height="180"> | <img src="diffused_3dsplat_image00.png" alt="diffused_3dsplat_image00" width="225" height="180"> |

###### Second test

- Select images from novel view with 0.4 probability
- Conclude the problems encountered: denoising steps are not enough; forget to normalize before diffusion and denormalize after that
- timesteps for adding noise: 50
- timesteps for denoise: 15
- Noise level: sample from standard normal distribution * 4
- Results:
  
| Non-diffused Image   | Diffused Image |
| ------------- | ------------ |
|  <img src="non-diffused_3dsplat_image01.png" alt="non_diffused_3dsplat_image01" width="225" height="180"> | <img src="diffused_3dsplat_image01.png" alt="diffused_3dsplat_image01" width="225" height="180"> |

##### Comparation between 3D Gaussian Splat and DDIM-Splat performance

###### Procedure

1. Run splatfacto(3D Gaussian Splat) for 10000 steps on original images.
2. Run ddim-splatfacto and splatfacto for another 500 steps
3. Evaluate the model with evaluation command offerd by nerfstudio

![image](https://github.com/user-attachments/assets/be639e28-4d45-403f-813b-1fb07759de81)

###### Analysis

1. The default evaluation is over original data that also used for train, so the Structural Similarity Index (SSIM) measuring the similarity between two images is with better scores for splatfacto trained on original data only. However, the Learned Perceptual Image Patch Similarity (LPIPS) , a metric used to evaluate the perceptual similarity between images with Lower values indicating higher similarity shows a lower score on DDIM Gaussian Splatting model.
2. num_rays_per_sec indicates the number of rays processed per second, which is a measure of rendering speed. Frames per second (FPS) is a measure of how many frames the system can render per second. Higher FPS indicates smoother performance. From the above plot of evaluation json file, we see a smoother performance and quicker rendering for DDIM Gaussian Splatting model.

###### Renderings

- Gaussian Splatting Rendering

[![Watch the video](https://github.com/user-attachments/assets/44a515d8-11ea-4a0d-b16c-770ea1dff707)](https://github.com/user-attachments/assets/44a515d8-11ea-4a0d-b16c-770ea1dff707)

- DDIM-Splat Rendering in the same trajectory

[![Watch the video](https://github.com/user-attachments/assets/57329a2d-026d-492f-9fd2-341feb7e10a9)](https://github.com/user-attachments/assets/57329a2d-026d-492f-9fd2-341feb7e10a9)

Demonstrated by two videos, clearly, Gaussian Splatting rendering has a smoother surface, but is constraint by viewing angles. For example, it shows more artifacts, and the scenes are blured. In comparison, although it is not so smooth for DDIM-Splat rendering, it presents more detail compared with the 3DGS only model. 

##### Later work

Refine the model by
1. either adding surface alignment techque in the pipeline
2. or save the diffused images that have low lpip value(more similar) and train with those data(disable remove gaussian, and enable if it is original data)

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week13.md)
