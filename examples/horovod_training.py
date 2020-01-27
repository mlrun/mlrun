from __future__ import print_function
import os
import sys
import json
import keras
from keras.applications.vgg16 import VGG16
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    Activation, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import horovod.keras as hvd
import pandas as pd
from sklearn.model_selection import train_test_split
import io

from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact

# Acquire MLRun context
mlctx = get_or_create_ctx('horovod-trainer')

# Get env variables
mlctx.logger.info('Getting env variables')
DATA_PATH = mlctx.get_input('data_path').url #, '/User/horovod-trainer/data/cats_n_dogs')
MODEL_PATH = mlctx.get_param('model_path', '/tmp/models/model.hd5')#, '/User/horovod-trainer/models/catsndogs.hd5')
CHECKPOINTS_DIR = mlctx.get_param('checkpoints_dir')#, '/User/horovod--trainer/checkpoints')

mlctx.logger.info(f'Validating paths:\n'\
                  f'Data_path:\t{DATA_PATH}\n'\
                  f'Model_path:\t{MODEL_PATH}\n')
#os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


categories_map = str(mlctx.get_input('categories_map').get())
mlctx.logger.info(f'Categories map: {categories_map}')
df = pd.read_csv(str(mlctx.get_input('file_categories')))


mlctx.logger.info(f'Got {df.shape[0]} files in {DATA_PATH}')
mlctx.logger.info(f'Training data has {df.size} samples')
mlctx.logger.info(f'{df.category.value_counts()}')

# Get image parameters
IMAGE_WIDTH = mlctx.get_param('image_width')#, 128)
IMAGE_HEIGHT = mlctx.get_param('image_height')#, 128)
IMAGE_CHANNELS = mlctx.get_param('image_channels')#, 3)  # RGB color
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

# Get training parameters
epochs = mlctx.get_param('epochs', 1)
batch_size = mlctx.get_param('batch_size', 64)

# Check for GPU
is_gpu_available = False
if tf.test.gpu_device_name():
    is_gpu_available = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
mlctx.logger.info(f'Is GPU available?\t{is_gpu_available}')


#
# Training
#


# Prepare, test, and train the data
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df['category'] = train_df['category'].astype('str');
validate_df['category'] = validate_df['category'].astype('str');
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process).
config = tf.ConfigProto()
if is_gpu_available:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

# load model
model = VGG16(include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

    # add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
output = Dense(1, activation='sigmoid')(class1)

# define new model
model = Model(inputs=model.inputs, outputs=output)

# Horovod: adjust learning rate based on number of GPUs.
# opt = keras.optimizers.SGD(lr=0.001, momentum=0.9)
opt = keras.optimizers.Adadelta(lr=1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    # Reduce the learning rate if training plateaues.
    keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint(
        os.path.join(CHECKPOINTS_DIR, 'checkpoint-{epoch}.h5')))

# Set up ImageDataGenerators to do data augmentation for the training images.
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_datagen.mean = [123.68, 116.779, 103.939]

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    DATA_PATH,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)
mlctx.logger.info(f'classes: {train_generator.class_indices}')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_datagen.mean = [123.68, 116.779, 103.939]
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    DATA_PATH,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)

# Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_train // batch_size,
    callbacks=callbacks,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=total_validate // batch_size
)

# save the model only on worker 0 to prevent failures ("cannot lock file")
if hvd.rank() == 0:
    MODEL_DIR = os.path.dirname(MODEL_PATH)
    model.save(MODEL_PATH)
    with open(os.path.join(MODEL_DIR, 'model-architecture.json'), 'w') as f:
        f.write(model.to_json())
    model.save_weights(os.path.join(MODEL_DIR, 'model-weights.h5'))
    mlctx.logger.info(f'history: {history.history}')
    mlctx.log_artifact('model', target_path=MODEL_PATH, labels={'framework': 'tensorflow'})
    
    chart = ChartArtifact('summary.html')
    chart.header = ['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss']
    for i in range(epochs):
        chart.add_row([i+1, history.history['accuracy'][i], 
                       history.history['val_accuracy'][i], 
                       history.history['loss'][i], 
                       history.history['val_loss'][i]])
    mlctx.log_artifact(chart)
    mlctx.log_result('loss', float(history.history['loss'][epochs-1]))
    mlctx.log_result('accuracy', float(history.history['accuracy'][epochs-1]))
