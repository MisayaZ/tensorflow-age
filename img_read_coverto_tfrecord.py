import tensorflow as tf
import numpy as np

import os
import cv2
#from PIL import Image
import glob
#import faceinitial


def dense_to_one_hot(labels_dense, num_classes=6):
  """
  Convert class labels from scalars to one-hot vectors.


  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  print(labels_one_hot[0])

  return labels_one_hot


def read_jpgs_from(path, num_classes, one_hot=False):
    """Reads directory of images.
    Args:
      path: path to the directory

    Returns:
      A list of all images in the directory in the TF format (You need to call sess.run() or .eval() to get the value).
    """
    filename = os.path.join('record_save', 'validation' + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    for age_i in range(num_classes):
        jpg_files_path = glob.glob(os.path.join(path + str(age_i), '*.[jJ][pP][gG]'))
        for filename in jpg_files_path:
            #im = Image.open(filename)
            print filename
            im = cv2.imread(filename)
            im = cv2.resize(im, (48,48))
            rows = im.shape[0]
            cols = im.shape[1]
            depth = im.shape[2]
            im = np.asarray(im, np.uint8)
            
            # split filename path to save label  ex: /home/brandon/cv/p/8_ (n).png  => 8
            label_name = filename.split('/')[-1].split('_')[0]

            image_raw = im.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                          'height': _int64_feature(rows),
                          'width': _int64_feature(cols),
                          'depth': _int64_feature(depth),
                          'label': _int64_feature(int(label_name)),
                          'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
    writer.close()


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(images, labels, name):
  num_examples = labels.shape[0]
  if images.shape[0] != num_examples:
    raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
  rows = images.shape[1]
  cols = images.shape[2]
  depth = images.shape[3]

  filename = os.path.join('record_save', name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    print index
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(labels[index])),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())



def main(argv):
  # Get the data.
  print('transfer image data to tfrecords')

  train_data_dir = '/home/bryan/data/Iris/age-train-file-2019.8/89-data/train-crop/'
  validation_data_dir = '/home/bryan/data/Iris/age-train-file-2019.8/89-data/test-crop/'

  num_classes = 89
  
  read_jpgs_from(validation_data_dir, num_classes, one_hot=False)
  #validation_images, validation_labels = read_jpgs_from(validataion_data_dir, num_classes, one_hot=False)	
  

  # Convert to Examples and write the result to TFRecords.
  #convert_to(train_images, train_labels, 'train') 
  #convert_to(validation_images, validation_labels, 'validation')
  

  print('Done')


if __name__ == '__main__':
  tf.app.run()

