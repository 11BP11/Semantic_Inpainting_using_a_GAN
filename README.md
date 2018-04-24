# Semantic_Inpainting_using_a_GAN
My attempt of using a GAN to do semantic inpainting for my Part III essay.

Code is based on the code from Taehoon Kim's DCGAN-tensorflow project:<br />
https://github.com/carpedm20/DCGAN-tensorflow###

The model trains the GAN to take an incomplete picture as the input in addition to z,
and use this to generate a picture including the given part.<br />

# Usage

To train a model one can e.g. use:<br />
$ python main.py --dataset mnist --input_height=28 --output_height=28 --train <br />
$ python main.py --dataset celebA --input_height=108 --crop --train

# Examples

![Example1](/good_samples/celebA_8e_merge.png) <br />
Samples after 8 epochs. Here the generator completed the picture receiving approximately the lower third of the picture. <br /> <br />

![Example2](/good_samples/celebA_8e_diffz_z_dim=4000_--img_height=12.png) <br />
Samples after 8 epochs. Here the variation with z is shown. Part supplied to the generator was marked for visualisation.<br /> <br />

