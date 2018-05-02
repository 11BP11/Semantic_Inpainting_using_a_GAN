import os
import sys
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
#Adam default, TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("generate_test_images", 100, "Number of images to generate during test. [100]")

flags.DEFINE_integer("vis_type", 0, "Visualization option; 0=all. [0]")
flags.DEFINE_integer("img_height", None, "Height of img given to G. If None, 4*floor(output_height/10) [None]")
flags.DEFINE_boolean("use_labels", False, "Whether to use labels. Only for mnist [False]")
flags.DEFINE_float("lambda_loss", 100., "Coefficient of L1-loss. [100.]")
flags.DEFINE_boolean("split_data", False, "Split data for Gen and Dist, further to between train and test. [False]")
flags.DEFINE_boolean("gen_use_img", False, "True for the generator using the input picture (img) as output. [False]")
flags.DEFINE_boolean("use_border", False, "True for using the top row throughout the generator. [False]")
flags.DEFINE_integer("z_dim", 100, "Dimension of the random input. [100]")

flags.DEFINE_boolean("drop_discriminator", False, "If True ignores the D and uses loss function instead. [False]")

FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if FLAGS.img_height is None:
    FLAGS.img_height = 4*int(FLAGS.output_height / 10)
    print("No img_hight supplied. img_height = %s" % FLAGS.img_height)
  
  if FLAGS.split_data is True:
    FLAGS.epoch = 2*FLAGS.epoch
    print("Number of epochs doubled to make it more comparable to version without split date")
    
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  #if not os.path.exists('samples_progress'):
  #  os.makedirs('samples_progress')
  for i in range(8):
    if not os.path.exists('samples_progress/part{:1d}'.format(i+1)):
      os.makedirs('samples_progress/part{:1d}'.format(i+1))
  
  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True
  

  with open('settings.txt', "w") as f:
    f.write("\n" + " ".join(sys.argv) + "\n\n")
    print("FLAGS values:")
    for key, val in flags.FLAGS.__flags.items():
      print(str([key, val]))
      f.write(str([key, val])+"\n")
    print()
      
  with tf.Session(config=run_config) as sess:
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          y_dim=10,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          img_height=FLAGS.img_height,
          use_labels=FLAGS.use_labels,
          lambda_loss=FLAGS.lambda_loss,
          split_data=FLAGS.split_data,
          gen_use_img=FLAGS.gen_use_img,
          drop_discriminator=FLAGS.drop_discriminator,
          use_border=FLAGS.use_border)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          sample_num=FLAGS.batch_size,
          z_dim=FLAGS.z_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          crop=FLAGS.crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          img_height=FLAGS.img_height,
          lambda_loss=FLAGS.lambda_loss,
          split_data=FLAGS.split_data,
          gen_use_img=FLAGS.gen_use_img,
          drop_discriminator=FLAGS.drop_discriminator,
          use_border=FLAGS.use_border)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    # Below is codes for visualization
    #OPTION = 1
    if FLAGS.vis_type == 0:
      vis_options = [6,7,9,10]
      for option in vis_options:
        print("Visualizing option %s" % option)
        OPTION = option
        visualize(sess, dcgan, FLAGS, OPTION)
    else:
      OPTION = FLAGS.vis_type
      visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
