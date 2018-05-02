"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import os
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size , dcgan.z_dim))
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, dcgan.z_dim - 1) for _ in xrange(dcgan.z_dim)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(dcgan.z_dim):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

  elif option == 6: #Original together with inpainting, coloured.
    #Prints: Half original + generated upper half
    #   and full generated next to full original
    batch_size = config.batch_size
    
    for idx in xrange(min(8,int(np.floor(1000/batch_size)))):
      print(" [*] %d" % idx)
      
      sample_inputs, sample_img, sample_labels = get_img(dcgan, idx*batch_size, batch_size, config.dataset, test=True)

      sample_z = np.random.uniform(-1, 1, size=(batch_size , dcgan.z_dim))
      
      if config.dataset == 'mnist' and config.use_labels:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img, dcgan.y: sample_labels})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img })

      col_img = colour_samples(samples, config.dataset, dcgan.img_height)
      col_input = colour_originals(sample_inputs, config.dataset)
      
      merged = np.concatenate((samples[:,:(dcgan.output_height-dcgan.img_height),:,:], \
                                  sample_inputs[:,(dcgan.output_height-dcgan.img_height):,:,:]),1)
      col_merged = colour_samples(merged, config.dataset, dcgan.img_height)
      
      if config.dataset == 'mnist':
        save_both(col_img, col_input, image_frame_dim, ('test_v6_compare_%s' % (idx)))
        save_images(col_merged, [image_frame_dim, image_frame_dim, 3], './samples/test_v6_merged_samples_%s.png' % (idx))
      else:
        save_both(samples, sample_inputs, image_frame_dim, ('test_v6_compare_%s' % (idx)))
        save_images(merged, [image_frame_dim, image_frame_dim, 3], './samples/test_v6_merged_samples_%s.png' % (idx))
  
  elif option == 7: #Save 4x4(6x6) image of merged samples (not coloured). 
    batch_size = config.batch_size
    
    for idx in xrange(min(8,int(np.floor(1000/batch_size)))):
      print(" [*] %d" % idx)
      
      sample_inputs, sample_img, sample_labels = get_img(dcgan, idx*batch_size, batch_size, config.dataset, test=True)

      sample_z = np.random.uniform(-1, 1, size=(batch_size , dcgan.z_dim))
      
      if config.dataset == 'mnist' and config.use_labels:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img, dcgan.y: sample_labels})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img })

      merged = np.concatenate((samples[:,:(dcgan.output_height-dcgan.img_height),:,:], \
                                  sample_inputs[:,(dcgan.output_height-dcgan.img_height):,:,:]),1)
            
      if config.dataset == 'mnist':
        merged_subset = merged[0:36]
        save_images(merged_subset, [6, 6, 3], './samples/test_v7_merged_samples_%s.png' % (idx))      
      else:
        merged_subset = merged[0:16]
        save_images(merged_subset, [4, 4, 3], './samples/test_v7_merged_samples_%s.png' % (idx))
            
  elif option == 8: #different values of z. Version to avoid batch normalization effect if this causes troubles"
    batch_size = config.batch_size
    length = int(np.sqrt(config.batch_size))
    
    sample_inputs0, sample_img0, sample_labels0 = get_img(dcgan, 0, batch_size, config.dataset, test=True)
    
    class_z = np.random.randint(2, size=dcgan.z_dim)
    values = np.linspace(-1., 1., num=length)
    z_values = np.empty((0,dcgan.z_dim))
      
    for i in range(length): #create z
      for j in range(length):
        z_values = np.append(z_values, [class_z * values[i] + (1-class_z) * values[j]], axis=0)
    
    shuff = np.zeros((0,batch_size)) #2nd column: permutations of 0:63
    for i in xrange(batch_size):
      x = np.arange(batch_size)
      random.shuffle(x)
      shuff = np.append(shuff, [x], axis=0).astype(int)
    
    all_samples = np.empty((batch_size,batch_size,dcgan.output_height,dcgan.output_width,dcgan.c_dim))
    
    for idx in xrange(batch_size): #over all noice variations.
      print(" [*] %d" % idx) 
    
      sample_inputs = sample_inputs0
      sample_labels = sample_labels0
      sample_img = sample_img0
            
      sample_z = np.zeros((batch_size,dcgan.z_dim))
      for i in range(batch_size):
        z = z_values[shuff[i,idx]]        
        sample_z[i,:] = z        
      
      if config.dataset == "mnist" and config.use_labels:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img, dcgan.y: sample_labels})     
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: sample_z, dcgan.img: sample_img })     
      
      for i in range(batch_size):
        all_samples[i,shuff[i,idx],:,:,:] = np.copy(samples[i])
      
    for idx in range(batch_size):
      
      samples = all_samples[idx,:,:,:,:]
      
      col_img = colour_samples(samples, config.dataset, dcgan.img_height)
      
      save_images(col_img, [image_frame_dim, image_frame_dim, 3], './samples/test_v8_diffz_%s.png' % (idx))
  
  elif option == 9: #different values of z.
    batch_size = config.batch_size
    length = int(np.sqrt(config.batch_size))
    
    sample_inputs0, sample_img0, sample_labels0 = get_img(dcgan, 0, batch_size, config.dataset, test=True)
    
    class_z = np.random.randint(2, size=dcgan.z_dim)
    values = np.linspace(-1., 1., num=length)
    z_values = np.empty((0,dcgan.z_dim))
      
    for i in range(length): #create z
      for j in range(length):
        z_values = np.append(z_values, [class_z * values[i] + (1-class_z) * values[j]], axis=0)
    
    for idx in range(64):
      print(" [*] %d" % idx)
    
      sample_inputs = np.repeat([sample_inputs0[idx]], batch_size, axis=0)
      if config.dataset == 'mnist':
        sample_labels = np.repeat([sample_labels0[idx]], batch_size, axis=0)
      else:
        sample_labels = None
      sample_img = np.repeat([sample_img0[idx]], batch_size, axis=0)
      
      if config.dataset == "mnist" and config.use_labels:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_values, dcgan.img: sample_img, dcgan.y: sample_labels})     
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_values, dcgan.img: sample_img })     
      
      col_img = colour_samples(samples, config.dataset, dcgan.img_height)
      
      save_images(col_img, [image_frame_dim, image_frame_dim, 3], './samples/test_v9_diffz_%s.png' % (idx))
  
  elif option == 10: #Take pictures from samples_progress and put them into one file.
    for i in range(8):
      prog_pics_base = glob(os.path.join('./samples_progress','part{:1d}'.format(i+1), '*.jpg'))    
    
      #prog_pics_base = glob(os.path.join('./samples_progress', '*.jpg'))
      imreadImg = imread(prog_pics_base[0])
      
      prog_pics = [
            get_image(prog_pic,
                      input_height=dcgan.output_height,
                      input_width=dcgan.output_height,
                      resize_height=dcgan.output_height,
                      resize_width=dcgan.output_width,
                      crop=dcgan.crop,
                      grayscale=dcgan.grayscale) for prog_pic in prog_pics_base]
      prog_pics_conv = np.array(prog_pics).astype(np.float32)   
      
      print(prog_pics_conv.shape)
      
      out_pics = np.reshape(prog_pics_conv, (64,prog_pics_conv.shape[1],prog_pics_conv.shape[2],-1))
      print(out_pics.shape)
      
      
      save_images(out_pics, [image_frame_dim, image_frame_dim], './samples_progress/progress{:1d}.png'.format(i+1))
      
  elif option == 11: #Save pictures centered and aligned in ./data_aligned
    
    if True: #training data
      if not os.path.exists('data_aligned'):
        os.makedirs('data_aligned')
        
      nr_samples = len(dcgan.data)
      batch_size = config.batch_size
      print(nr_samples)
      print(batch_size)
      
      batch_idxs = nr_samples // batch_size
      for idx in range(batch_idxs):
        sample_inputs, _, _ = get_img(dcgan, idx*batch_size, batch_size, config.dataset, test=False)  
        for i in range(batch_size):
          pic_idx = idx*batch_size + i
          save_images(sample_inputs[i:i+1:], [1,1],
                          './data_aligned/al{:06d}.jpg'.format(pic_idx+1))
        print("Done [%s] out of [%s]" % (idx,batch_idxs))
    
    if True: #test data   
      if not os.path.exists('data_test_aligned'):
        os.makedirs('data_test_aligned')
      nr_samples = 1000    
      sample_inputs, _, _ = get_img(dcgan, 0, nr_samples, config.dataset, test=True)
      for pic_idx in range(nr_samples):
        save_images(sample_inputs[pic_idx:pic_idx+1:], [1,1],
                        './data_test_aligned/aligned{:03d}.jpg'.format(pic_idx+1))
      
    
def get_img(dcgan, start_idx, batch_size, dataset, test=True, type=''):
    
    if dataset == 'mnist':
      if test:
        sample_inputs = dcgan.data_X_test[start_idx:(start_idx+batch_size)]
        sample_labels = dcgan.data_y_test[start_idx:(start_idx+batch_size)]
      else:
        if type == 'gen':
          sample_inputs = dcgan.data_X_gen[start_idx:(start_idx+batch_size)]
          sample_labels = dcgan.data_y_gen[start_idx:(start_idx+batch_size)]
        elif type == 'dis':
          sample_inputs = dcgan.data_X_dis[start_idx:(start_idx+batch_size)]
          sample_labels = dcgan.data_y_dis[start_idx:(start_idx+batch_size)]
        else:
          sample_inputs = dcgan.data_X[start_idx:(start_idx+batch_size)]
          sample_labels = dcgan.data_y[start_idx:(start_idx+batch_size)]

    else:
      sample_labels = None
      if test:
        sample_files = dcgan.data_test[start_idx:(start_idx+batch_size)]
      else:
        if type == 'gen':
          sample_files = dcgan.data_gen[start_idx:(start_idx+batch_size)]
        elif type == 'dis':
          sample_files = dcgan.data_dis[start_idx:(start_idx+batch_size)]
        else:
          sample_files = dcgan.data[start_idx:(start_idx+batch_size)]
      sample = [
          get_image(sample_file,
                    input_height=dcgan.input_height,
                    input_width=dcgan.input_width,
                    resize_height=dcgan.output_height,
                    resize_width=dcgan.output_width,
                    crop=dcgan.crop,
                    grayscale=dcgan.grayscale) for sample_file in sample_files]
      if (dcgan.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
      
    sample_half_images = sample_inputs[:,(dcgan.output_height-dcgan.img_height):,:,:]
    sample_img = np.reshape(sample_half_images, (batch_size , dcgan.img_dim))
    
    return sample_inputs, sample_img, sample_labels
      
def colour_samples(samples, dataset, height):

  if dataset == "mnist":  
    input_half = samples[:,(samples.shape[1]-height):,:,0]
    input_half_zeros = np.zeros_like(input_half)
    input_half_col = np.stack((input_half_zeros,input_half_zeros,input_half), -1)
    
    generated_half = samples[:,:(samples.shape[1]-height),:,0]
    #generated_half_zeros = np.zeros_like(generated_half)
    generated_half_col = np.stack((generated_half,generated_half,generated_half), -1) 
  else:
    input_half = samples[:,(samples.shape[1]-height):,:,:]
    input_half_col = input_half * 0.5
    
    generated_half = samples[:,:(samples.shape[1]-height),:,:]
    generated_half_col = generated_half 
   
  col_img = np.concatenate((generated_half_col,input_half_col),1)
  
  return col_img
  
def colour_originals(originals, dataset):
  if dataset == "mnist":
    originals_zeros = np.zeros_like(originals[:,:,:,0])
    col_input = np.stack((originals_zeros,originals[:,:,:,0],originals_zeros), -1)
  else:
    col_input = originals
  
  return col_input
  
def save_both(col_img, col_input, image_frame_dim, name):
  batch_size = col_img.shape[0]
  output = np.empty_like(col_img)
  
  output[::2] = col_img[:int(batch_size / 2):]
  output[1::2] = col_input[:int(batch_size / 2):]
  
  save_images(output, [image_frame_dim, image_frame_dim], './samples/' + name + 'a.png' ) #3?

  output[::2] = col_img[int(batch_size / 2)::]
  output[1::2] = col_input[int(batch_size / 2)::]
  
  save_images(output, [image_frame_dim, image_frame_dim, 3], './samples/' + name + 'b.png' )


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
