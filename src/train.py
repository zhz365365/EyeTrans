import tensorflow as tf
import argparse
import input_dataset
import os
import numpy as np
import math
import time
import random
import cv2
import GD_Net as GD_Net

slim = tf.contrib.slim

parser = argparse.ArgumentParser(description='')
parser.add_argument('--GPU', default='0', help='the index of gpu')
parser.add_argument('--batch_size', default=8, help='the size of examples in per batch')
parser.add_argument('--epoch', default=20, help='the train epoch')
parser.add_argument('--lr_boundaries', default='5,10,15', help='the boundaries of learning rate')
parser.add_argument('--lr_values', default='0.0001,0.00001,0.000001,0.0000001', help='the values of learning rate')
parser.add_argument('--data_dir', default='../dataset/', help='the directory of trainging data')
parser.add_argument('--server', default='77', help='server')
parser.add_argument('--image_height', default='160', help='the height of image')
parser.add_argument('--image_width', default='256', help='the width of image')
parser.add_argument('--dropout_keep_prob', default=0.5, help='the probility to keep dropout')
parser.add_argument('--trainable', default='0', help='train or not')
parser.add_argument('--stack', default='1', help='the number of stacked hourglasses')
parser.add_argument('--use_batch_norm_G', default='1', help='whether or not use BN in G')
parser.add_argument('--use_batch_norm_D', default='0', help='whether or not use BN in D')
parser.add_argument('--method_of_GAN', default='DCGAN', help='use DCGAN or WGAN')
args = parser.parse_args()
args.dropout_keep_prob = float(args.dropout_keep_prob)
args.use_batch_norm_G = bool(args.use_batch_norm_G == '1')
args.use_batch_norm_D = bool(args.use_batch_norm_D == '1')
args.stack = int(args.stack)
args.batch_size = int(args.batch_size)
args.image_height = int(args.image_height)
args.image_width = int(args.image_width)
args.epoch = float(args.epoch)

lr_boundaries = [float(key) for key in args.lr_boundaries.split(',')]
lr_values = [float(key) for key in args.lr_values.split(',')]
summary_dir = '../Log/'+args.server+'/event_log'
restore_dir = '../Log/'+args.server+'/check_log'
draw_dir 	= '../Log/'+args.server+'/draw_log'
checkpoint_dir = '../Log/'+args.server+'/check_log/model.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

def g_parameter(checkpoint_exclude_scopes):
	exclusions = []
	if checkpoint_exclude_scopes:
		exclusions = [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
	variables_to_restore = []
	variables_to_train = []
	for var in slim.get_model_variables():
		excluded = False
		for exclusion in exclusions:
			if var.op.name.startswith(exclusion):
				excluded = True
				variables_to_train.append(var)
				print(var.op.name)
				break
		if not excluded:
			variables_to_restore.append(var)
	return variables_to_restore, variables_to_train

class compute(object):
	def __init__(self, Net, image_syn, image_real, image_fake, G_epoch,
				 real, fake, is_training, dropout_keep_prob, smooth=0.1):
		self.Net = Net
		self.image_syn = image_syn
		self.image_fake = image_fake
		self.image_real = image_real
		self.G_epoch = G_epoch
		self.real = real
		self.fake = fake
		self.is_training = is_training
		self.dropout_keep_prob = dropout_keep_prob
		self.smooth = smooth
		self._loss()
	def _loss(self):
		if args.method_of_GAN == 'DCGAN':
			self.D_loss_real = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.real,
					labels=tf.ones_like(self.real)*(1-self.smooth)
				)
			)
			self.D_loss_fake = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.fake,
					labels=tf.zeros_like(self.fake)
				)
			)
			self.G_loss_fake = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.fake,
					labels=tf.ones_like(self.fake)
				)
			)
			self.ratio = 1e-2
		elif args.method_of_GAN.startswith('WGAN'):
			self.D_loss_real = tf.reduce_mean(
				-self.real
			)
			self.D_loss_fake = tf.reduce_mean(
				self.fake
			)
			self.G_loss_fake = tf.reduce_mean(
				-self.fake
			)
			self.ratio = 1e-2
			if args.method_of_GAN == 'WGAN-gp':
				alpha = tf.random_uniform(shape=[args.batch_size, 1, 1, 1],minval=0.,maxval=1.)
				interpolates = alpha*self.image_fake + (1-alpha)*self.image_real
				bias = self.Net.build_D(
					interpolates,
					is_training=self.is_training,
					dropout_keep_prob=self.dropout_keep_prob,
					reuse=True)
				gradients = tf.gradients(bias, [interpolates])[0]
				slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
				gradient_penalty = tf.reduce_mean((slopes-1.)**2)
				lamda = 10
				self.D_loss_penalty = lamda * gradient_penalty
		elif args.method_of_GAN == 'BEGAN':
			self.image_fake = tf.clip_by_value(self.image_fake, 0, 1/255.0)
			self.real = tf.clip_by_value(self.real, 0, 1/255.0)
			self.fake = tf.clip_by_value(self.fake, 0, 1/255.0)
			self.D_loss_real = tf.reduce_mean(
				tf.abs(self.real - self.image_real)
			)
			self.D_loss_fake = tf.reduce_mean(
				-tf.abs(self.fake - self.image_fake)
			)
			self.G_loss_fake = tf.reduce_mean(
				tf.abs(self.fake - self.image_fake)
			)
			self.k_t = tf.Variable(0., trainable=False, name='k_t')
			self.D_loss_fake *= self.k_t
			self.gamma = 0.5
			self.lambda_k = 0.001
			self.balance = self.gamma * self.D_loss_real - self.G_loss_fake
			self.measure = self.D_loss_real + tf.abs(self.balance)
			self.ratio = 1e-2
		else:
			raise Exception('wrong method of GAN!')
		self.D_loss = self.D_loss_real + self.D_loss_fake
		self.G_loss_reg = self.ratio*tf.reduce_sum(
			tf.reduce_mean(
				tf.reduce_sum(tf.abs(self.image_syn-self.image_fake), axis=[1, 2]), axis=[0, 1]
			)
		)		
		if args.method_of_GAN == 'BEGAN':
			self.G_loss = self.G_loss_fake
		elif args.method_of_GAN.startswith('WGAN') or args.method_of_GAN == 'DCGAN':
			self.G_loss = tf.where(
				tf.less(0.5, self.G_epoch),
				self.G_loss_fake + self.G_loss_reg,
				self.G_loss_reg
			)
		else:
			raise Exception('wrong method of GAN!')
		if args.method_of_GAN == 'WGAN-gp':
			self.D_loss += self.D_loss_penalty

def visual_generator(syn, fake, epoch):
	num = random.randint(0, args.batch_size-1)
	syn = (syn[num,:,:,0]+1)/2.0*255.0
	fake = cv2.resize((fake[num,:,:,0]+1)/2.0*255.0, None, fx=4, fy=4)
	cv2.imwrite(draw_dir+'/%.4f_syn.png'%epoch, syn)
	cv2.imwrite(draw_dir+'/%.4f_fake.png'%epoch, fake)

def train():
	print(tf.__version__)
	with tf.Graph().as_default():
		Data = input_dataset.Dataset_reader(
			batch_size=args.batch_size,
			image_height=args.image_height,
			image_width=args.image_width
		)
		image_syn = tf.placeholder(
			tf.float32,
			[args.batch_size, args.image_height, args.image_width, 1]
		)
		image_real = tf.placeholder(
			tf.float32,
			[args.batch_size, args.image_height//4, args.image_width//4, 1]
		)
		dropout_keep_prob = tf.placeholder(tf.float32)
		is_training = tf.placeholder(tf.bool)
	
		Net = GD_Net.GD_Net(
			use_batch_norm_G=args.use_batch_norm_G,
			use_batch_norm_D=args.use_batch_norm_D,
		)
		image_fake, image_syn_res = Net.build_G(
			image_syn,
			is_training=is_training,
			dropout_keep_prob=dropout_keep_prob,
			reuse=False,
		)
		if args.method_of_GAN.startswith('WGAN') or args.method_of_GAN == 'DCGAN':
			D_image_real = Net.build_D(
				image_real,
				is_training=is_training,
				dropout_keep_prob=dropout_keep_prob,
				reuse=False,
			)
			D_image_fake = Net.build_D(
				image_fake,
				is_training=is_training,
				dropout_keep_prob=dropout_keep_prob,
				reuse=True,
			)
		elif args.method_of_GAN == 'BEGAN':
			D_image_real = Net.build_auto_encoder_decoder(
				image_real,
				is_training=is_training,
				dropout_keep_prob=dropout_keep_prob,
				reuse=False,
			)
			D_image_fake = Net.build_auto_encoder_decoder(
				image_fake,
				is_training=is_training,
				dropout_keep_prob=dropout_keep_prob,
				reuse=True,
			)
		else:
			raise Exception('wrong method of GAN!')

		global_step_G = tf.get_variable('global_step_G', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
		num_epoch_G = tf.get_variable('num_epoch_G', [], trainable=False, dtype=tf.float32)
		num_epoch_G = tf.cast(global_step_G, tf.float32) * args.batch_size / (Data.G_train_num)
		lr_G = tf.train.piecewise_constant(num_epoch_G, boundaries=lr_boundaries, values=lr_values)

		global_step_D = tf.get_variable('global_step_D', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int64)
		num_epoch_D = tf.get_variable('num_epoch_D', [], trainable=False, dtype=tf.float32)
		num_epoch_D = tf.cast(global_step_D, tf.float32) * args.batch_size / (Data.D_train_num)
		lr_D = tf.train.piecewise_constant(num_epoch_D, boundaries=lr_boundaries, values=lr_values)

		g = compute(Net, image_syn_res, image_real, image_fake, num_epoch_G, D_image_real, D_image_fake, is_training, dropout_keep_prob)

		for key in Net.G_end_points.keys():
			print(key, ' ', Net.G_end_points[key].get_shape().as_list())
		for key in Net.D_end_points.keys():
			print(key, ' ', Net.D_end_points[key].get_shape().as_list())

		variables_to_restore_G, variables_to_train_G = g_parameter(Net.G_scope)
		variables_to_restore_D, variables_to_train_D = g_parameter(Net.D_scope)

		with tf.name_scope('D_loss') as D_loss_scope:
			tf.summary.scalar('D_loss_real', g.D_loss_real)
			tf.summary.scalar('D_loss_fake', g.D_loss_fake)
			if args.method_of_GAN == 'WGAN-gp':
				tf.summary.scalar('D_loss_penalty', g.D_loss_penalty)
			tf.summary.scalar('D_loss', g.D_loss)
			D_loss_merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, D_loss_scope))
		with tf.name_scope('G_loss') as G_loss_scope:
			tf.summary.scalar('G_loss_reg', g.G_loss_reg)
			tf.summary.scalar('G_loss_fake', g.G_loss_fake)
			tf.summary.scalar('G_loss', g.G_loss)
			G_loss_merged = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, G_loss_scope))
		
		var_list_G = variables_to_train_G
		var_list_D = variables_to_train_D
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies([tf.group(*update_ops)]):
			"""
			G_train_op = tf.train.AdamOptimizer(learning_rate=lr_G)\
				.minimize(g.G_loss, var_list=var_list_G, global_step=global_step_G, name='Adam_G')
			"""
			G_train_op = tf.train.GradientDescentOptimizer(learning_rate=lr_G)\
				.minimize(g.G_loss, var_list=var_list_G, global_step=global_step_G, name='Adam_G')
			"""
			G_train_op = tf.train.RMSPropOptimizer(learning_rate=lr_G)\
				.minimize(g.G_loss, var_list=var_list_G, global_step=global_step_G, name='Adam_G')
			D_train_op = tf.train.AdamOptimizer(learning_rate=lr_D)\
				.minimize(g.D_loss, var_list=var_list_D, global_step=global_step_D, name='Adam_D')
			"""
			D_train_op = tf.train.GradientDescentOptimizer(learning_rate=lr_D)\
				.minimize(g.D_loss, var_list=var_list_D, global_step=global_step_D, name='Adam_D')
			"""
			D_train_op = tf.train.RMSPropOptimizer(learning_rate=lr_D)\
				.minimize(g.D_loss, var_list=var_list_D, global_step=global_step_D, name='Adam_D')
			"""
			D_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in var_list_D]
		if args.method_of_GAN == 'BEGAN':
			with tf.control_dependencies([G_train_op, D_train_op]):
				k_update = tf.assign(
                	g.k_t, tf.clip_by_value(g.k_t + g.lambda_k * g.balance, 0, 1)
				)

		saver_list = tf.global_variables()
		init = tf.global_variables_initializer()
		saver_restore = tf.train.Saver(saver_list)
		saver_train = tf.train.Saver(saver_list, max_to_keep=100)

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
											  log_device_placement=False)) as sess:
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)
			summary_writer_train_G = tf.summary.FileWriter(logdir=summary_dir+'/train_G')
			summary_writer_train_D = tf.summary.FileWriter(logdir=summary_dir+'/train_D')
			summary_writer_eval_G = tf.summary.FileWriter(logdir=summary_dir+'/eval_G')
			summary_writer_eval_D = tf.summary.FileWriter(logdir=summary_dir+'/eval_D')

			ckpt = tf.train.get_checkpoint_state(restore_dir)
			if ckpt and ckpt.model_checkpoint_path:
				temp_dir = ckpt.model_checkpoint_path
				temp_step = int(temp_dir.split('-')[1])
				print('Restore the global parameters in the step %d!' % temp_step)
				saver_restore.restore(sess, temp_dir)
			else:
				print('Initialize the global parameters!')
				init.run()
			
			eval_epoch = 0
			checkpoint_epoch = 0
			if args.method_of_GAN == 'DCGAN':
				Kg = 50
				Kr = 1
			elif args.method_of_GAN.startswith('WGAN'):
				Kg = 50
				Kr = 5
			if args.trainable=='1': print('Begin the training!')
			while(args.trainable=='1'):
				"""
				Train
				""" 
				if args.method_of_GAN.startswith('WGAN') or args.method_of_GAN == 'DCGAN':
					for i in range(Kr):
						image_syn_batch, image_real_batch = sess.run([Data.G_train_batch[0], Data.D_train_batch[0]])
						feed_dict_train = {
							image_syn:image_syn_batch,
							image_real:image_real_batch,
							is_training:True,
							dropout_keep_prob:args.dropout_keep_prob
						}
						start_time = time.time()
						if args.method_of_GAN == 'WGAN':
						 	sess.run(D_clip)
						D_summary, D_loss_value, D_step, D_epoch, _ = sess.run(
							[D_loss_merged, g.D_loss, global_step_D, num_epoch_D, D_train_op],
							feed_dict=feed_dict_train
						)
						end_time = time.time()
						sec_per_batch = end_time-start_time
						examples_per_sec = float(args.batch_size)/sec_per_batch
						summary_writer_train_D.add_summary(D_summary, int(1000*D_epoch))
						print('Discriminator epoch %.3f step %d with loss %.8f(%.3f examples/sec; %.3f sec/batch)'\
							  %(D_epoch, D_step, D_loss_value, examples_per_sec, sec_per_batch))
					for i in range(Kg):
						image_syn_batch = sess.run(Data.G_train_batch[0])
						feed_dict_train = {
							image_syn:image_syn_batch,
							is_training:True,
							dropout_keep_prob:args.dropout_keep_prob
						}
						start_time = time.time()
						G_summary, G_loss_value, G_step, G_epoch, _ = sess.run(
							[G_loss_merged, g.G_loss, global_step_G, num_epoch_G, G_train_op],
							feed_dict=feed_dict_train
						)
						end_time = time.time()
						sec_per_batch = end_time-start_time
						examples_per_sec = float(args.batch_size)/sec_per_batch
						summary_writer_train_G.add_summary(G_summary, int(1000*G_epoch))
						print('Generator epoch %.3f step %d with loss %.8f(%.3f examples/sec; %.3f sec/batch)'\
							  %(G_epoch, G_step, G_loss_value, examples_per_sec, sec_per_batch))
				elif args.method_of_GAN == 'BEGAN':
					image_syn_batch, image_real_batch = sess.run([Data.G_train_batch[0], Data.D_train_batch[0]])
					feed_dict_train = {
						image_syn:image_syn_batch,
						image_real:image_real_batch,
						is_training:True,
						dropout_keep_prob:args.dropout_keep_prob
					}
					start_time = time.time()
					D_summary, D_loss_value, D_step, D_epoch, G_summary, G_loss_value, G_step, G_epoch, k_update_value, measure_value = sess.run(
						[D_loss_merged, g.D_loss, global_step_D, num_epoch_D, G_loss_merged, g.G_loss, global_step_G, num_epoch_G, k_update, g.measure],
						feed_dict=feed_dict_train
					)
					end_time = time.time()
					sec_per_batch = end_time-start_time
					examples_per_sec = float(args.batch_size)/sec_per_batch
					summary_writer_train_D.add_summary(D_summary, int(1000*D_epoch))
					summary_writer_train_G.add_summary(G_summary, int(1000*G_epoch))
					print('Discriminator epoch %.3f step %d with loss %.8f; k_update %.6f; measure %.6f(%.3f examples/sec; %.3f sec/batch)'\
						  %(D_epoch, D_step, D_loss_value, k_update_value, measure_value, examples_per_sec, sec_per_batch))
					print('Generator epoch %.3f step %d with loss %.8f(%.3f examples/sec; %.3f sec/batch)'\
						  %(G_epoch, G_step, G_loss_value, examples_per_sec, sec_per_batch))
				else:
					raise Exception('wrong method of GAN!')
				"""
				Eval & visual generator pictures
				"""
				if G_epoch >= eval_epoch + 0.0005:
					image_syn_batch = sess.run(Data.G_eval_batch[0])
					feed_dict_eval = {
						image_syn:image_syn_batch,
						is_training:False,
						dropout_keep_prob:1.0
					}
					G_summary, image_fake_batch = sess.run(
						[G_loss_merged, image_fake],
						feed_dict=feed_dict_eval
					)
					visual_generator(image_syn_batch, image_fake_batch, G_epoch)
					summary_writer_eval_G.add_summary(G_summary, int(1000*G_epoch))
					eval_epoch = G_epoch
				"""
				save model
				"""
				if G_epoch >= checkpoint_epoch + 0.1:
					saver_train.save(sess, checkpoint_dir, global_step=G_step)
					checkpoint_epoch = G_epoch
				"""
				finish the training
				"""
				if G_epoch >= args.epoch:
					print('Finish the training!')
					break
				if G_epoch >= 0.5:
					Kg = 2

			summary_writer_train_G.close()
			summary_writer_train_D.close()
			summary_writer_eval_G.close()
			summary_writer_eval_D.close()
			coord.request_stop()
			coord.join(threads)

if __name__ == '__main__':
	print('----------------------------------train.py start-----------------------------------------------')
	train()
