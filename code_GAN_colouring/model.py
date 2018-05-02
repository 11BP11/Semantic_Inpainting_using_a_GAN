from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         #y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None,
         lambda_loss=100):
    
    #Need:  -> tform function, same function in tf
    #       -> scale_to (some function f such that: x' = f(x', G(x',z)' )'      , for ' denoting the tform function. 
    
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num
    
    self.lambda_loss = lambda_loss    
    
    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.z_dim = z_dim
    
    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.dataset_name == 'mnist':
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.g_bn3 = batch_norm(name='g_bn3')

    data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))
    imreadImg = imread(data[0])
    if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
      self.c_dim = imread(data[0]).shape[-1]
    else:
      self.c_dim = 1
    self.data = data[1000:]
    self.data_test = data[:1000]    
      
    self.g_input_dim = output_width*output_width*1
    
    self.grayscale = (self.c_dim == 1)
    
    self.build_model()

  def build_model(self):
    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.g_input = tf.placeholder(
      tf.float32, [None, self.g_input_dim], name='g_input')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.g_input)
    self.D, self.D_logits   = self.discriminator(self.inputs, reuse=False)
    #self.sampler            = self.sampler(self.z, self.img, self.y)
    self.sampler            = self.generator(self.z, self.g_input, sampler=True)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
    
    def L1_loss(x_img, img):
      v_length = x_img.get_shape().as_list()[1]
      return tf.scalar_mul(1./max(v_length,1), tf.norm(x_img - img, ord=1, axis=-1))
      #return tf.scalar_mul(1./tf.maximum(v_length,1), tf.norm(x_img - img, ord=1, axis=-1))
    def L2_loss(x_img, img):
      v_length = x_img.get_shape().as_list()[1]
      return tf.scalar_mul(1./v_length, tf.norm(x_img - img, ord=2, axis=-1))
    def full_L1_loss(input, output, batch_size): #
      in_vec = tf.reshape(input, [batch_size, -1])
      out_vec = tf.reshape(output, [batch_size, -1])
      v_length = in_vec.get_shape().as_list()[1]
      return tf.scalar_mul(1./tf.maximum(v_length,1), tf.norm(in_vec - out_vec, ord=1, axis=-1))    
    def additional_loss(x_dash, g_dash):
      v_length = x_dash.get_shape().as_list()[1]
      return tf.scalar_mul(1./max(v_length,1), tf.norm(x_dash - g_dash, ord=1, axis=-1))
      #return tf.scalar_mul(1./tf.maximum(v_length,1), tf.norm(x_img - img, ord=1, axis=-1))
    
    #Same transformation in TF:
    self.G_tformed = tf.reduce_mean(self.G, 3)
    self.G_tf_flat = tf.reshape(self.G_tformed, (self.batch_size, self.g_input_dim))
    
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D))) # = -log(sigmoid( D_logits ))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_))) # = -log(1 - sigmoid( D_logits_ ))
    self.g_loss1 = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) # = -log(sigmoid( D_logits_ ))
    self.g_loss2 = tf.scalar_mul(self.lambda_loss, tf.reduce_mean(
      additional_loss(self.G_tf_flat, self.g_input)))
      #L1_loss(self.G_img, self.img)))
    self.g_loss = self.g_loss1 + self.g_loss2
    
    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
              
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    sample_inputs = get_img(self, 0, self.sample_num, config.dataset, test=False)
    test_inputs = get_img(self, 0, self.sample_num, config.dataset, test=True)
    
    sample_g_input = np.reshape(self.tform(sample_inputs), (self.batch_size , self.g_input_dim))
    test_g_input = np.reshape(self.tform(test_inputs), (self.batch_size , self.g_input_dim))
    
    save_images(sample_inputs, image_manifold_size(sample_inputs.shape[0]),'samples\original_image_sample.png')
    save_images(test_inputs, image_manifold_size(sample_inputs.shape[0]),'samples\original_image_test.png')

    save_images(self.tform(sample_inputs), image_manifold_size(sample_inputs.shape[0]),'samples\original_input_sample.png')
    save_images(self.tform(test_inputs), image_manifold_size(sample_inputs.shape[0]),'samples\original_input_test.png')
    
    #Set up for visualizing difference from z value
    side_length = int(np.sqrt(config.batch_size))
    class_z = np.random.randint(2, size=self.z_dim)
    values = np.linspace(-1., 1., num=side_length)
    z_range = np.empty((0,self.z_dim))
      
    for i in range(side_length): #create z
      for j in range(side_length):
        z_range = np.append(z_range, [class_z * values[i] + (1-class_z) * values[j]], axis=0)
        
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    avg_g_loss = 0
        
    for epoch in xrange(config.epoch):
      #self.data = glob(os.path.join(
      #  "./data", config.dataset, self.input_fname_pattern))
      data_length = len(self.data)
      batch_idxs = min(data_length, config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        batch_images = get_img(self, idx*config.batch_size, \
              config.batch_size, config.dataset, test=False)
        
        batch_g_input = np.reshape(self.tform(batch_images), (self.batch_size , self.g_input_dim))
        
        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        D_dict = {
            self.inputs: batch_images,
            self.z: batch_z,
            self.g_input: batch_g_input}
        G_dict = {
            self.z: batch_z, 
            self.g_input: batch_g_input}
          
        PreerrG = self.g_loss.eval(G_dict)
        
        # Update D network
        _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict=D_dict)
        self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=G_dict)
        self.writer.add_summary(summary_str, counter)

        # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
        _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict=G_dict)
        self.writer.add_summary(summary_str, counter)
          
        errD_fake = self.d_loss_fake.eval(D_dict)
        errD_real = self.d_loss_real.eval(D_dict)
        errG1 = self.g_loss1.eval(G_dict)
        errG2 = self.g_loss2.eval(G_dict)
        errG = self.g_loss.eval(G_dict)
        #errG = errG1 + errG2

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.4f, g_loss: %.4f (%.2f)" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, PreerrG))
        #Should hold: errD <= log(4) ~ 1.39 (= error for random guessing)
        
        outp_c = 20
        if np.mod(counter, outp_c) == 1:
          print("g_loss: %.8f (D) + %.8f (input) = %.8f" % (errG1, errG2, errG))

          sample_dict={ #For generator and discriminator
                self.z: sample_z,
                self.g_input: sample_g_input,
                self.inputs: sample_inputs,
          }
          test_dict={
                self.z: sample_z,
                self.g_input: test_g_input,
                self.inputs: test_inputs,
          }
          
          try:
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss], feed_dict=sample_dict)
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            
            test_samples, test_d_loss, test_g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss], feed_dict=test_dict)
            save_images(test_samples, image_manifold_size(test_samples.shape[0]),
                  './{}/train_{:02d}_{:04d}_test.png'.format(config.sample_dir, epoch, idx))
            print("[Test] d_loss: %.8f, g_loss: %.8f" % (test_d_loss, test_g_loss)) 
          except:
            print("one pic error!...")

        if np.mod(counter, 5*outp_c) == 1:
          self.save(config.checkpoint_dir, counter)
          
          image_frame_dim = int(math.ceil(config.batch_size**.5))
          
          c_idx = np.mod(counter, batch_idxs)
          c_epoch = int(np.floor(counter/batch_idxs))
          
          #col_img = colour_samples(samples, config.dataset, self.img_height)
          #col_input = colour_originals(sample_inputs, config.dataset)
          save_both(samples, sample_inputs, image_frame_dim, ('train_{:02d}_{:04d}_col'.format(epoch, idx)))
          save_four(sample_inputs, self.tform(sample_inputs), samples, self.tform(samples), \
                      image_frame_dim, ('train_{:02d}_{:04d}_ovw'.format(epoch, idx)))
          
          #col_img = colour_samples(test_samples, config.dataset, self.img_height)
          #col_input = colour_originals(test_inputs, config.dataset)
          save_both(test_samples, test_inputs, image_frame_dim, ('train_{:02d}_{:04d}_col_test'.format(epoch, idx)))
          save_four(test_inputs, self.tform(test_inputs), test_samples, self.tform(test_samples), \
                      image_frame_dim, ('train_{:02d}_{:04d}_ovw_test'.format(epoch, idx)))
          
          print("Checkpoint!")
          
          #Visualize change with z:
          print("visualizing for different z values ...")
          for i in range(2):          
            input_idx = np.random.randint(self.batch_size)
            
            vis_z_g_input = np.repeat([test_g_input[input_idx]],self.batch_size,axis=0)
            vis_z_inputs = np.repeat([test_inputs[input_idx]],self.batch_size,axis=0)
            
            vis_z_dict={
                self.z: z_range,
                self.g_input: vis_z_g_input,
                self.inputs: vis_z_inputs,
            }
         
            samples = self.sess.run(self.sampler, feed_dict=vis_z_dict)

            save_images(samples, image_manifold_size(test_samples.shape[0]),
                  './{}/train_{:02d}_{:04d}_vis_z_{:01d}.png'.format(config.sample_dir, epoch, idx, input_idx))
      
      else: #(=Do, if loop finishes without error)
        #Visualize at the end of every epoch
        if epoch<8:
          for j in range(8):
            for i in range(8):
              pic_idx = 8*j + i
              save_images(test_samples[pic_idx:pic_idx+1:], [1,1],
                      './samples_progress/part{:01d}/pic{:02d}_epoch{:02d}.jpg'.format(j+1, pic_idx, epoch))
              
    #save a final checkpoint
    #self.save(config.checkpoint_dir, counter)
      
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4
 
  def generator(self, z, g_input, y=None, sampler=False):
    with tf.variable_scope("generator") as scope:
      if sampler:
        scope.reuse_variables()
      do_train = not sampler
      do_with_w = not sampler
      
      #if not self.y_dim:
      if not self.dataset_name == 'mnist':
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        
        # merge z and img to one vector:
        z = concat([z, g_input], 1) #length: 100 + 64*64*1 = 100 + 4069 
        #old length: 100 + 24*64*3 = 100 + 4608
        
        if not sampler: 
          # project `z` and reshape
          self.z_, self.h0_w, self.h0_b = linear(
              z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=do_with_w)
              
          self.h0 = tf.reshape(
              self.z_, [self.batch_size, s_h16, s_w16, self.gf_dim * 8])
          h0 = tf.nn.relu(self.g_bn0(self.h0, train=do_train))
          
          self.h1, self.h1_w, self.h1_b = deconv2d(
              h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=do_with_w)
          h1 = tf.nn.relu(self.g_bn1(self.h1, train=do_train))
          
          h2, self.h2_w, self.h2_b = deconv2d(
              h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=do_with_w)
          h2 = tf.nn.relu(self.g_bn2(h2, train=do_train))
          
          h3, self.h3_w, self.h3_b = deconv2d(
              h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=do_with_w)
          h3 = tf.nn.relu(self.g_bn3(h3, train=do_train))
          
          h4, self.h4_w, self.h4_b = deconv2d(
              h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=do_with_w)
        else:
          # project `z` and reshape
          h0 = tf.reshape(
              linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
              [self.batch_size, s_h16, s_w16, self.gf_dim * 8])
              #[-1, s_h16, s_w16, self.gf_dim * 8])
          h0 = tf.nn.relu(self.g_bn0(h0, train=False))
          
          h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
          h1 = tf.nn.relu(self.g_bn1(h1, train=False))
          
          h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
          h2 = tf.nn.relu(self.g_bn2(h2, train=False))
          
          h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
          h3 = tf.nn.relu(self.g_bn3(h3, train=False))
          
          h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')      
        
        return tf.nn.tanh(h4) #values in (-1,1) for celebA
  
  def tform(self, img_batch): #Transformation we want to do the inverse of.
    #dims = img_batch.shape
    #batch_size = dims[0]
    bw_img = bw(img_batch)
    #bw_flat = np.reshape(bw_img, (batch_size , -1))
    bw_flat = np.reshape(bw_img, (self.batch_size , self.g_input_dim))
    #return bw_flat
    #return bw_img.reshape(self.batch_size,self.output_width,self.output_height,1)
    return bw_img
  
  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    #return X/255.,y_vec
    return X[1000:-1]/255., y_vec[1000:-1], X[0:1000]/255., y_vec[0:1000] #why not map to (-1,1)?
    
  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
