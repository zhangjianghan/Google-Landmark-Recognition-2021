import os

# os.system('pip install -q tensorflow~=2.2.0 tensorflow_gcs_config~=2.2.0')
os.system('pip install -q efficientnet')
# os.system('pip install -q gcsfs')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import numpy as np
import pandas as pd
import random
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import efficientnet.tfkeras as efn
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tqdm.notebook import tqdm as tqdm
import requests
from kaggle_datasets import KaggleDatasets

# resp = requests.post("http://{}:8475/requestversion/{}".format(os.environ["COLAB_TPU_ADDR"].split(":")[0], tf.__version__))
# if resp.status_code != 200:
#    print("Failed to switch the TPU to TF {}".format(version))

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

#
# Data access
GCS_PATH = KaggleDatasets().get_gcs_path(
    'landmark-tfrecords-384')  # 'gs://kds-3873db5988cab73fb3f61387f4260e23cbff4012248d13a9fa5dd335'
# KaggleDatasets().get_gcs_path('landmark-tfrecords-384')
GCS_PATH_2 = KaggleDatasets().get_gcs_path(
    'landmark-tfrecords-384-2')  # 'gs://kds-e44a20ea6436cd0271b97c5b431450bcd70a463206a983e1acb88874'
# KaggleDatasets().get_gcs_path('landmark-tfrecords-384-2')
DICT_PATH = KaggleDatasets().get_gcs_path('landmark-image-train') + '/train_encoded.csv'
# KaggleDatasets().get_gcs_path('landmark-image-train')

# Configuration
EPOCHS = 20
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [384, 384]
# Seed
SEED = 100
# Learning rate
LR = 0.0001
# Number of classes
NUMBER_OF_CLASSES = 81313
VALID_STAGE = False
# Training filenames directory
files1 = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')
files2 = tf.io.gfile.glob(GCS_PATH_2 + '/train*.tfrec')
assert len(files1) > 0
assert len(files2) > 0
FILENAMES = files1 + files2
# Read csv file
df = pd.read_csv(DICT_PATH)
# Using 20% of the data to validate
TRAINING_FILENAMES, VALIDATION_FILENAMES = train_test_split(FILENAMES, test_size=0.10, random_state=SEED)
training_groups = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in TRAINING_FILENAMES]
validation_groups = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in VALIDATION_FILENAMES]
n_trn_classes = df[df['group'].isin(training_groups)]['landmark_id_encode'].nunique()
n_val_classes = df[df['group'].isin(validation_groups)]['landmark_id_encode'].nunique()
print(f'The number of unique training classes is {n_trn_classes} of {NUMBER_OF_CLASSES} total classes')
print(f'The number of unique validation classes is {n_val_classes} of {NUMBER_OF_CLASSES} total classes')


# Seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


# Function to decode our images (normalize and reshape)
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    # Convert image to floats in [0, 1] range
    # image = tf.cast(image, tf.float32) #/ 255.0
    # Explicit size needed for TPU
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


# This function parse our images and also get the target variable
def read_tfrecord(example):
    TFREC_FORMAT = {
        # tf.string means bytestring
        "image": tf.io.FixedLenFeature([], tf.string),
        # shape [] means single element
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'])
    target = tf.cast(example['target'], tf.int32)
    return image, target


# This function load our tf records and parse our data with the previous function
def load_dataset(filenames, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    ignore_order = tf.data.Options()
    if not ordered:
        # Disable order, increase speed
        ignore_order.experimental_deterministic = False

        # Automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # Use data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    # Returns a dataset of (image, label) pairs
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset


# This function output the data so that we can use arcface
def arcface_format(image, target):
    image = tf.cast(image, tf.float32) / 255.0
    return {'inp1': image, 'inp2': target}, target


# augment

def augment(image, target):
    image = tf.cast(image, tf.uint8)
    image = tf.image.random_flip_left_right(image)
    image = tf.cast(image, tf.float32)
    return image, target


# Training data pipeline
def get_training_dataset(filenames, ordered=False):
    dataset = load_dataset(filenames, ordered=ordered)
    dataset = dataset.map(augment, num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)
    # The training dataset must repeat for several epochs
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    # Prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO)
    return dataset


# Validation data pipeline
def get_validation_dataset(filenames, ordered=True, prediction=False):
    dataset = load_dataset(filenames, ordered=ordered)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)
    # If we are in prediction mode, use bigger batch size for faster prediction
    if prediction:
        dataset = dataset.batch(BATCH_SIZE * 4)
    else:
        dataset = dataset.batch(BATCH_SIZE)
    # Prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO)
    return dataset


# Count the number of observations with the tabular csv
def count_data_items(filenames):
    records = [int(re.compile(r"_([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    df = pd.read_csv(DICT_PATH)
    n = df[df['group'].isin(records)].shape[0]
    return n


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
print(f'Training with {NUM_TRAINING_IMAGES} images')
print(f'Validating with {NUM_VALIDATION_IMAGES} images')

tf.keras.utils.get_custom_objects()['ArcMarginProduct'] = ArcMarginProduct


# Function to build our model using fine tunning (efficientnet)
def get_model():
    with strategy.scope():
        margin = ArcMarginProduct(
            n_classes=NUMBER_OF_CLASSES,
            s=64,
            m=0.2,
            name='head/arc_margin',
            dtype='float32'
        )

        inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')
        label = tf.keras.layers.Input(shape=(), name='inp2')
        x = efn.EfficientNetB5(weights='imagenet', include_top=False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(2048)(x)
        # x = tf.keras.layers.Dense(2048)(x)
        # x = tf.keras.layers.Dense(2048)(x)
        x = margin([x, label])

        output = tf.keras.layers.Softmax(dtype='float32')(x)
        model = tf.keras.models.Model(inputs=[inp, label], outputs=[output])
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

        model.compile(
            optimizer=opt,
            loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        return model


# Seed everything
seed_everything(SEED)

# Build training and validation generators
train_dataset = get_training_dataset(TRAINING_FILENAMES, ordered=False)
val_dataset = get_validation_dataset(VALIDATION_FILENAMES, ordered=True, prediction=False)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
filepath = f'b5.h5'
# Using a checkpoint to save best model (want the entire model, not only the weights)
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor='val_loss',
                                                save_best_only=True,
                                                save_weights_only=False)
# Using learning rate scheduler
cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      mode='min',
                                                      factor=0.5,
                                                      patience=1,
                                                      verbose=1,
                                                      min_delta=0.0001)

model = get_model()
# Train and evaluate our model
history = model.fit(train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    callbacks=[cb_lr_schedule, checkpoint],
                    validation_data=val_dataset,
                    verbose=2)