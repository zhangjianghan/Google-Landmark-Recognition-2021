import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import pathlib
from tqdm import tqdm


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2):
    feature = {
        'id': _bytes_feature(feature0),
        'image': _bytes_feature(feature1),
        'target': _int64_feature(feature2)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


TRAIN_IMAGE_DIR = '../input/landmark-recognition-2021/train'
TRAIN = '../input/landmark-image-train/train_encoded.csv'


# Read image and resize it
def read_image(image_path, size=(384, 384)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img


def get_tf_records(record=0, size=(384, 384)):
    df = pd.read_csv(TRAIN)
    # Get image paths
    image_paths = [x for x in pathlib.Path(TRAIN_IMAGE_DIR).rglob('*.jpg')]
    # Get only one group, this is a slow process so we need to make 50 different sessions
    df = df[df['group'] == record]
    # Reset index
    df.reset_index(drop=True, inplace=True)
    # Get a list of ids
    ids_list = list(df['id'].unique())
    # Write tf records
    with tf.io.TFRecordWriter('train_{}.tfrec'.format(record)) as writer:
        for image_path in tqdm(image_paths):
            image_id = image_path.name.split('.')[0]
            if image_id in ids_list:
                # Get target
                target = df[df['id'] == image_id]['landmark_id_encode']
                img = read_image(str(image_path), size)
                img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 100))[1].tostring()
                example = serialize_example(
                    str.encode(image_id), img, target.values[0]
                )
                writer.write(example)

# get_tf_records(record = 0, size = (384, 384))