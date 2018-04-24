# Semantic_Inpainting_using_a_GAN
My attempt of using a GAN to do semantic inpainting.

Code is based on the code from Taehoon Kim's DCGAN-tensorflow project:
https://github.com/carpedm20/DCGAN-tensorflow###

To train a model one can e.g. use:
$ python main.py --dataset mnist --input_height=28 --output_height=28 --train
$ python main.py --dataset celebA --input_height=108 --crop --train
