----------
###### Title: 2024 Robotics and Computation Dissertation - Week 11
###### Date: 5-08-2024 -- 11-08-2024
----------
###### Monday-Sunday

#### Problems encountered during coding ddim_splat

1. Shape of trained DDIM model does not fit 3DGS
   Sol:resize to (256 x 256) for DDIM and interpolate to recover the size for rendering
2. It is worse to train with diffusion model, when perturbing both gaussian means and rendered image,
   and train with completely novel views.
   Sol: give up the perturb on gaussian means, and train like Dyna in RL: with a certain probability,
   the model will train with original view images, while in other cases, it uses the images rendered from model
   and processes it with diffusion model.


#### Write Background of the project:

##### Useful website:

###### 3D Gaussian Splatting:

- [3D Gaussian Splatting 原理略解-知乎](https://zhuanlan.zhihu.com/p/675326584)
- [[Paper Review] 3D Gaussian Splatting (SIGGRAPH 2023): Improving Rendering Speed/Quality](https://xoft.tistory.com/51)
- [[Concept Summary] 3D Gaussian and 2D projection](https://xoft.tistory.com/49)
- [3D Gaussian Splatting中的数学推导](https://zhuanlan.zhihu.com/p/666465701)

###### Diffusion

- [Denoising Diffusion Implicit Models,DDIM](https://www.zhangzhenhu.com/aigc/ddim.html#equation-eq-ddim-216)
- [Diffusion Probabilistic Models](https://www.zhangzhenhu.com/aigc/%E6%89%A9%E6%95%A3%E6%A6%82%E7%8E%87%E6%A8%A1%E5%9E%8B.html#equation-eq-ddpm-039)

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week12.md)
