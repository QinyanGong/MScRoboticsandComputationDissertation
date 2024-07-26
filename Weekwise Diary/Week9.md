----------
###### Title: 2024 Robotics and Computation Dissertation - Week 8
###### Date: 8-07-2024 -- 13-07-2024
----------
###### Monday-Sunday

#### Train Diffusion models : Denoising Diffusion Probabilistic Model (DDPM) & Denoising Diffusion Implicit Model (DDIM)

- Use DDPM & DDIM pipeline, scheduler, and UNet models in diffusier library for training a denoising diffusion model

#### The result for DDPM & DDIM denoising noise image twice

###### Output images processed by DDPM from sampled noise with 10 inference timesteps 

| First Image   | Second Image |
| ------------- | ------------ |
|  <img src="ddpm_noise_10step_output(0).png" alt="ddpm_noise_10step_output(0)" width="250" height="250"> | <img src="ddpm_noise_10step_output(1).png" alt="ddpm_noise_10step_output(1)" width="250" height="250"> |


##### Output images processed by DDIM from sampled noise with 10 inference timesteps

|First Image|Second Image|
|:-:|:-:|
|<img src="ddim_noise_10step_output(0).png" alt="ddim_noise_10step_output(0)" width="250" height="250">|<img src="ddim_noise_10step_output(1).png" alt="ddim_noise_10step_output(1)" width="250" height="250">|

##### Output images processed by DDPM & DDIM from noisy sample image
<img src="very_noisy_sample.png" alt="very_noisy_sample" width="250" height="250">

|12 Steps DDPM|100 Steps DDPM|12 Steps DDIM|
|:-:|:-:|:-:|
|<img src="denoise_very_noisy_sample_ddpm_12steps.png" alt="denoise_very_noisy_sample_ddpm_12steps" width="250" height="250">|<img src="denoise_very_noisy_sample_ddpm_100steps.png" alt="denoise_very_noisy_sample_ddpm_100steps" width="250" height="250">|<img src="denoise_very_noisy_sample_ddim_12steps.png" alt="denoise_very_noisy_sample_ddim_12steps" width="250" height="250">|


#### Conclusion:
- The DDPM pipeline is adapt to generate image from pure noise while DDIM from images being added noise on
- The reason could be DDIM captures the non-Markovian properties between x0 and xt: x0 is not just affected by x1 but also by xt (the noisy image we provide).
- Therefore, I will choose DDIM pipeline to be integrated into GSDiffusion model

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week9.md)
