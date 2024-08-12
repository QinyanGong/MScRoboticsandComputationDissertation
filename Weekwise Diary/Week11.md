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

&nbsp;
----------
&nbsp;
> ###### [Next Week](Week12.md)
