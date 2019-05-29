from PIL import Image
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import cv2 as cv
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'

class Dataset_reader(object):
	def __init__(self, batch_size, image_height=160, image_width=256, min_queue_examples=512):
		self.batch_size = batch_size
		self.image_height = image_height
		self.image_width = image_width
		self.image_size = [image_height, image_width]
		self.min_queue_examples = min_queue_examples
		self.G_train_num = 960000
		self.G_eval_num = 240000
		self.D_train_num = 213658
		self._dataset_build()

	def _dataset_build(self):
		if os.path.exists('../dataset/UnityEyes/train.tfrecords'):
			self.G_train_batch = self._G_tfrecords_to_batch(
				[['../dataset/UnityEyes/train.tfrecords']],
				'G_train',
			)
		else:
			raise Exception('there is no G train data!')

		if os.path.exists('../dataset/UnityEyes/eval.tfrecords'):
			self.G_eval_batch = self._G_tfrecords_to_batch(
				[['../dataset/UnityEyes/eval.tfrecords']],
				'G_eval',
			)
		else:
			raise Exception('there is no G eval data!')
		if os.path.exists('../dataset/MPIIGaze/train.tfrecords'):
			self.D_train_batch = self._D_tfrecords_to_batch(
				[['../dataset/MPIIGaze/train.tfrecords']],
				'D_train',
			)
		else:
			raise Exception('there is no D train data!')

	def _G_tfrecords_to_batch(self, tfrecords_list, name):
		example_list = [self._read_G_image(i) for i in tfrecords_list]
		img_batch, label_batch, ldmks_batch = tf.train.shuffle_batch_join(
			example_list,
			batch_size=self.batch_size,
			capacity=self.min_queue_examples+3*self.batch_size,
			min_after_dequeue=self.min_queue_examples,
			name=name
		)
		return (img_batch, label_batch, ldmks_batch)

	def _D_tfrecords_to_batch(self, tfrecords_list, name):
		example_list = [self._read_D_image(i) for i in tfrecords_list]
		img_batch, label_batch = tf.train.shuffle_batch_join(
			example_list,
			batch_size=self.batch_size,
			capacity=self.min_queue_examples+3*self.batch_size,
			min_after_dequeue=self.min_queue_examples,
			name=name
		)
		return (img_batch, label_batch)

	def _read_G_image(self, filename):
		filename_queue = tf.train.string_input_producer(filename)
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
			serialized_example, 
			features={
				'img':tf.FixedLenFeature([], tf.string),
				'ldmks_interior_margin':tf.FixedLenFeature([], tf.string),
				'ldmks_caruncle':tf.FixedLenFeature([], tf.string),
				'ldmks_iris':tf.FixedLenFeature([], tf.string),
			}
		)
		img = tf.decode_raw(features['img'], tf.uint8)
		img = tf.reshape(img, [300, 400, 3])
		img = tf.cast(img, tf.float32)

		img_center = tf.constant([150, 200], dtype=tf.float32)

		ldmks_interior_margin = tf.decode_raw(features['ldmks_interior_margin'], tf.float64)
		ldmks_interior_margin = tf.reshape(ldmks_interior_margin, [16, 2])
		ldmks_interior_margin = tf.cast(ldmks_interior_margin, tf.float32)

		ldmks_caruncle = tf.decode_raw(features['ldmks_caruncle'], tf.float64)
		ldmks_caruncle = tf.reshape(ldmks_caruncle, [7, 2])
		ldmks_caruncle = tf.cast(ldmks_caruncle, tf.float32)

		ldmks_iris = tf.decode_raw(features['ldmks_iris'], tf.float64)
		ldmks_iris = tf.reshape(ldmks_iris, [32, 2])
		ldmks_iris = tf.cast(ldmks_iris, tf.float32)

		ldmks = tf.concat([ldmks_interior_margin, ldmks_caruncle, ldmks_iris], axis=0)
		ldmks = tf.reshape(ldmks, [55, 2])

		return self._G_augmentation(img, img_center, ldmks)

	def _read_D_image(self, filename):
		filename_queue = tf.train.string_input_producer(filename)
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
			serialized_example,
			features={
				'img_raw':tf.FixedLenFeature([], tf.string),
				'label_raw':tf.FixedLenFeature([], tf.string),
			}
		)
		img_raw = tf.decode_raw(features['img_raw'], tf.uint8)
		img_raw = tf.reshape(img_raw, [self.image_height//4, self.image_width//4, 3])
		img_raw = tf.cast(img_raw, tf.float32)

		label_raw = tf.decode_raw(features['label_raw'], tf.float64)
		label_raw = tf.reshape(label_raw, [2])
		label_raw = tf.cast(label_raw, tf.float32)

		return self._D_augmentation(img_raw, label_raw)

	def _G_augmentation(self, img, img_center, ldmks):
		# res_map to roi_map
		def crop(img, img_center, img_size, ldmks):
			img_roi = img[int(img_center[0]-img_size[0]//2):int(img_center[0]+img_size[0]//2),
						  int(img_center[1]-img_size[1]//2):int(img_center[1]+img_size[1]//2),
						  :]
			img_bias = np.array([[img_center[1]-img_size[1]//2, img_center[0]-img_size[0]//2]])
			ldmks -= np.repeat(img_bias, np.size(ldmks, 0), axis=0)
			def clip(ldmks, size):
				return np.concatenate(
					(np.clip(ldmks[:,0], 0, size[1]-1)[:,np.newaxis], np.clip(ldmks[:,1], 0, size[0]-1)[:,np.newaxis]), 
					axis=1,
				)
			ldmks = clip(ldmks, img_size)
			return img_roi, ldmks
		"""
		augmentation
		"""
		"""
		# normal function
		def normal(size):
			norm = np.random.normal(loc=0.0, scale=1.0, size=size)
			norm = np.asarray(norm, dtype=np.float32)
			return norm
		# translation
		translation = 10*tf.py_func(normal, [2], tf.float32)
		img_center -= translation
		# scale
		scale = 1 + 0.05*normal(1)
		shape = img.get_shape().as_list()[0:2]
		scale_shape = [int(shape[0]*scale), int(shape[1]*scale)]
		img = tf.image.resize_images(img, scale_shape)
		img_center *= scale
		ldmks *= scale
		# down_scale_and_up_scale
		scale = math.exp(normal(1))
		old_shape = img.get_shape().as_list()[0:2]
		new_shape = [int(old_shape[0]*scale), int(old_shape[1]*scale)]
		img = tf.image.resize_images(img, new_shape)
		img = tf.image.resize_images(img, old_shape)
		# rotation有点难写 先空着
		# intensity tf的这个函数需要接受uint8类型的255数值 否则对于float32类型的数值会认为范围在0~1而不是0~255.0
		img = tf.cast(img, tf.uint8)
		img = tf.image.random_brightness(img, 50/255.)
		img = tf.cast(img, tf.float32)
		# blur
		def gauss(img, sigma):
			kernel_size = (7, 7)
			return cv2.GaussianBlur(img, kernel_size, sigma);
		sigma = 1 + 0.05*normal(1)
		img = tf.py_func(gauss, [img, sigma], tf.float32)
		"""
		# crop
		img_roi, ldmks = tf.py_func(
			crop, 
			[img, img_center, self.image_size, ldmks],
			[tf.float32, tf.float32]
		)
		labels = tf.py_func(self._ldmks_to_labels, [ldmks], tf.float32)
		labels = tf.reshape(labels, [self.image_height, self.image_width, 55])
		img_roi = 0.114*img_roi[:,:,0] + 0.587*img_roi[:,:,1] + 0.299*img_roi[:,:,2]
		img_roi = (img_roi/255.0)*2-1
		img_roi = tf.reshape(img_roi, [self.image_height, self.image_width, 1])
		ldmks = tf.reshape(ldmks, [55, 2])
		return img_roi, labels, ldmks

	def _D_augmentation(self, img_raw, label_raw):
		img_raw = 0.114*img_raw[:,:,0] + 0.587*img_raw[:,:,1] + 0.299*img_raw[:,:,2]
		img_raw = (img_raw/255.0)*2-1
		img_raw = tf.reshape(img_raw, [self.image_height//4, self.image_width//4, 1])
		return img_raw, label_raw

	def _ldmks_to_labels(self, ldmks):
		n = ldmks.shape[0]
		labels = np.zeros((self.image_height, self.image_width, n), dtype=np.float32)
		for i in range(n):
			labels[:,:,i] = self._makeGasssian(center=(ldmks[i,0], ldmks[i,1]))
		return labels

	def _makeGasssian(self, center, sigma=3):
		x = np.arange(0, self.image_width, 1, np.float32)
		y = np.arange(0, self.image_height, 1, np.float32)[:, np.newaxis]
		x0 = center[0]
		y0 = center[1]
		return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def test_input_dataset():
	Data = Dataset_reader(batch_size=8)
	with tf.Session() as sess:
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		img_G = sess.run(
			Data.G_train_batch[0],
		)
		for i in range(Data.batch_size):
			cv.imwrite('../test/G_%d.png'%i, ((img_G[i,:,:,0]+1)/2.0*255.0).astype('uint8'))
		img_D = sess.run(
			Data.D_train_batch[0],
		)
		for i in range(Data.batch_size):
			cv.imwrite('../test/D_%d.png'%i, ((img_D[i,:,:,0]+1)/2.0*255.0).astype('uint8'))
		coord.request_stop()
		coord.join(threads)

if __name__ == '__main__':
	test_input_dataset()
