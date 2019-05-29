from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim

class GD_Net(object):
	def __init__(self, use_batch_norm_G, use_batch_norm_D, features=64):
		self.features = features
		self.use_batch_norm_G = use_batch_norm_G
		self.use_batch_norm_D = use_batch_norm_D
		self.G_scope = 'Generator'
		self.D_scope = 'Discriminator'
	
	def activation_fn(self, x, leak=0.2, name="lrelu"):
		with tf.variable_scope(name):
			return tf.maximum(x, leak * x)

	def normalizer_fn(self, x, epsilon=1e-5):
		with tf.variable_scope('instance_norm'):
			mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
			scale = tf.get_variable(name='scale',
									shape=[x.get_shape()[-1]],
									initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
			offset = tf.get_variable(name='offset',
									 shape=[x.get_shape()[-1]],
									 initializer=tf.constant_initializer(0.0))
			out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
			return out

	def _D_arg_scope(self, weight_decay=0.00001):
		with slim.arg_scope(
			[slim.conv2d],
			padding='SAME',
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_regularizer=slim.l2_regularizer(weight_decay),
			activation_fn=self.activation_fn,
			normalizer_fn=self.normalizer_fn if self.use_batch_norm_D else None):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc

	def _G_arg_scope(self, weight_decay=0.00001):
		with slim.arg_scope(
			[slim.conv2d],
			padding='SAME',
			weights_regularizer=slim.l2_regularizer(weight_decay),
			biases_regularizer=slim.l2_regularizer(weight_decay),
			activation_fn=self.activation_fn,
			normalizer_fn=self.normalizer_fn if self.use_batch_norm_G else None):
			with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
				return arg_sc

	@slim.add_arg_scope
	def _bottleneck_v2(self, inputs, stride=1, use_batch_norm=True, features=None, outputs_collections=None, scope=None):
		with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
			depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
			if use_batch_norm:
				inputs = self.normalizer_fn(inputs)
			preact = self.activation_fn(inputs)
			if features == None:
				features = self.features
			shortcut = slim.conv2d(preact, features, [1, 1], stride=stride,
								   normalizer_fn=None, activation_fn=None,
								   scope='shortcut')
			residual = slim.conv2d(preact, features//4, [1, 1], stride=1, scope='conv1')
			residual = slim.conv2d(residual, features//4, 3, stride=stride, scope='conv2')
			residual = slim.conv2d(residual, features, [1, 1], stride=1,
								   normalizer_fn=None, activation_fn=None,
								   scope='conv3')
			output = shortcut + residual
			return slim.utils.collect_named_outputs(outputs_collections,
													sc.name,
													output)

	@slim.add_arg_scope
	def _up_add(self, a, b, outputs_collections=None, scope=None):
		with tf.variable_scope(scope, 'up_add', [a, b]) as sc:
			up_a = tf.image.resize_images(a, b.get_shape().as_list()[1:3], method=1)
			output = up_a + b
			return slim.utils.collect_named_outputs(outputs_collections,
													sc.name,
													output)

	def build_G(self, inputs, is_training, dropout_keep_prob, reuse):
		with slim.arg_scope(self._G_arg_scope()):
			with tf.variable_scope(self.G_scope, 'Generator', [inputs], reuse=False) as sc:
				end_points_collection = sc.original_name_scope + '_end_point'
				with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d,
									self._bottleneck_v2, self._up_add],
									outputs_collections=end_points_collection):
					with slim.arg_scope([self._bottleneck_v2], use_batch_norm=self.use_batch_norm_G):
						net = inputs
						with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
							# 160x256x3
							net = slim.conv2d(net, self.features//2, 3, stride=2, scope='conv1')
							# 80x128x256
							net = slim.conv2d(net, self.features, 1, stride=1, scope='conv2')
							net = slim.conv2d(net, self.features, 1, stride=1, scope='conv3')
						net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
						# 40x64x512
						"""
						hourglasses
						"""
						"""
						C1 = slim.utils.collect_named_outputs(end_points_collection, sc.name+'/C1', net)
						C1_fix = self._bottleneck_v2(C1, scope='C1_fix')
						C2 = self._bottleneck_v2(C1, stride=2, scope='C2')
						C2_fix = self._bottleneck_v2(C2, scope='C2_fix')
						C3 = self._bottleneck_v2(C2, stride=2, scope='C3')
						C3_fix = self._bottleneck_v2(C3, scope='C3_fix')
						C4 = self._bottleneck_v2(C3, stride=2, scope='C4')
						C4a = self._bottleneck_v2(C4, scope='C4a')
						C4b = self._bottleneck_v2(C4a, scope='C4b')
						C4c = self._bottleneck_v2(C4b, scope='C4c')
						C3b = self._up_add(C4c, C3_fix)
						C3c = self._bottleneck_v2(C3b, scope='C3c')
						C2b = self._up_add(C3c, C2_fix)
						C2c = self._bottleneck_v2(C2b, scope='C2c')
						C1b = self._up_add(C2c, C1_fix)
						C1c = self._bottleneck_v2(C1b, scope='C1c')
						if not self.use_batch_norm_G:
							C1c = slim.dropout(C1c, dropout_keep_prob, scope='dropout')
						net = C1c
						"""
						"""
						Residual-3
						"""
						for i in range(1, 4):
							net = self._bottleneck_v2(net, features=self.features, scope='Block1_%d'%i)
						generator_out = slim.conv2d(net, 1, [1, 1], stride=1,
													activation_fn=tf.nn.tanh, normalizer_fn=None, scope='out')
						shape = inputs.get_shape().as_list()
						shape = [shape[1]//4, shape[2]//4]
						generator_res = tf.image.resize_images(inputs, shape, method=1)
						self.G_end_points = slim.utils.convert_collection_to_dict(end_points_collection)
						return generator_out, generator_res

	def build_D(self, inputs, is_training, dropout_keep_prob, reuse):
		with slim.arg_scope(self._D_arg_scope()):
			with tf.variable_scope(self.D_scope, 'Discriminator', [inputs], reuse=reuse) as sc:
				end_points_collection = sc.original_name_scope + '_end_point'
				with slim.arg_scope([slim.conv2d, slim.max_pool2d, self._bottleneck_v2],
									outputs_collections=end_points_collection):
					with slim.arg_scope([self._bottleneck_v2], use_batch_norm=self.use_batch_norm_D):
						net = inputs
						# 40x64x3
						"""
						fully-convolution
						"""
						net = slim.conv2d(net, self.features//16, 3, stride=2, scope='conv1')
						# 20x32x8
						net = slim.conv2d(net, self.features//8, 3, stride=2, scope='conv2')
						# 10x16x16
						net = slim.conv2d(net, self.features//4, 1, stride=1, scope='conv3')
						# 10x16x32
						net = slim.conv2d(net, self.features//2, 3, stride=2, scope='conv4')
						# 5x8x64
						net = slim.conv2d(net, self.features, 1, stride=1, scope='conv5')
						# 5x8x128
						"""
						ResNet-26
						"""
						"""
						with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
							net = slim.conv2d(net, self.features//16, 3, stride=1, scope='conv1')
							net = slim.conv2d(net, self.features//8, 1, stride=1, scope='conv2')
							net = slim.conv2d(net, self.features//8, 1, stride=1, scope='conv3')
						# 40x64xfeatures/8
						for i in range(1, 3):
							net = self._bottleneck_v2(net, features=self.features//8, scope='Block1_%d'%i)
						net = self._bottleneck_v2(net, features=self.features//4, stride=2, scope='Block1')
						# 20x32xfeatures/4
						for i in range(1, 3):
							net = self._bottleneck_v2(net, features=self.features//4, scope='Block2_%d'%i)
						net = self._bottleneck_v2(net, features=self.features//2, stride=2, scope='Block2')
						# 10x16xfeatures/2
						for i in range(1, 3):
							net = self._bottleneck_v2(net, features=self.features//2, scope='Block3_%d'%i)
						net = self._bottleneck_v2(net, features=self.features, stride=2, scope='Block3')
						# 5x8xfeatures
						for i in range(1, 3):
							net = self._bottleneck_v2(net, features=self.features, scope='Block4_%d'%i)
						net = self._bottleneck_v2(net, features=self.features, stride=2, scope='Block4')
						# 3x4xfeatures
						"""
						if self.use_batch_norm_D:
							net = self.normalizer_fn(net)
							net = self.activation_fn(net)
						else:
							net = self.activation_fn(net)
							net = slim.dropout(net, dropout_keep_prob, scope='dropout1')
						net = slim.conv2d(net, 1, [1, 1], stride=1, padding='VALID',
										  activation_fn=None, normalizer_fn=None, scope='fc1')
						# hxwx1
						batch_size = net.get_shape().as_list()[0]
						net = tf.reshape(net, [batch_size, -1])
						# hw
						self.D_end_points = slim.utils.convert_collection_to_dict(end_points_collection)
						return net
	
	def build_auto_encoder_decoder(self, inputs, is_training, dropout_keep_prob, reuse):
		with slim.arg_scope(self._D_arg_scope()):
			with tf.variable_scope(self.D_scope, 'Discriminator', [inputs], reuse=reuse) as sc:
				end_points_collection = sc.original_name_scope + '_end_point'
				with slim.arg_scope([slim.conv2d, slim.max_pool2d, self._bottleneck_v2],
									outputs_collections=end_points_collection):
					with slim.arg_scope([self._bottleneck_v2], use_batch_norm=self.use_batch_norm_D):
						net = inputs
						# 40x64x3
						"""
						fully-convolution
						"""
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv1')
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv2')
						# 40x64x128
						net = slim.conv2d(net, self.features*2, 3, stride=2, scope='conv3')
						# 20x32x256
						net = slim.conv2d(net, self.features*2, 3, stride=1, scope='conv4')
						net = slim.conv2d(net, self.features*2, 3, stride=1, scope='conv5')
						# 20x32x256
						net = slim.conv2d(net, self.features*3, 3, stride=2, scope='conv6')
						# 10x16x384
						net = slim.conv2d(net, self.features*3, 3, stride=1, scope='conv7')
						net = slim.conv2d(net, self.features*3, 3, stride=1, scope='conv8')
						# 10x16x384
						net = slim.conv2d(net, self.features*4, 3, stride=2, scope='conv9')
						# 5x8x512
						net = slim.conv2d(net, self.features*4, 3, stride=1, scope='conv10')
						net = slim.conv2d(net, self.features*4, 3, stride=1, scope='conv11')
						# 5x8x512
						net = slim.conv2d(net, self.features//2, [5, 8], stride=1, padding='VALID',
										  activation_fn=None, normalizer_fn=None, scope='fc1')
						# 1x1x64
						net = tf.squeeze(net, [1, 2])
						# 64
						net = slim.fully_connected(net, 5*8*self.features, 
												   activation_fn=None, normalizer_fn=None, scope='fc2')
						# 5*8*128
						net = tf.reshape(net, [-1, 5, 8, self.features])
						# 5x8x128
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv12')
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv13')
						# 5x8x128
						net = tf.image.resize_images(net, [10, 16])
						# 10x16x128
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv14')
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv15')
						# 10x16x128
						net = tf.image.resize_images(net, [20, 32])
						# 20x32x128
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv16')
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv17')
						# 20x32x128
						net = tf.image.resize_images(net, [40, 64])
						# 40x64x128
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv18')
						net = slim.conv2d(net, self.features, 3, stride=1, scope='conv19')
						# 40x64x128
						net = slim.conv2d(net, 1, 1, stride=1,
										  activation_fn=None, normalizer_fn=None, scope='conv20')
						# 40x64x1
						self.D_end_points = slim.utils.convert_collection_to_dict(end_points_collection)
						return net
GD_Net.default_image_height=160
GD_Net.default_image_width=256
