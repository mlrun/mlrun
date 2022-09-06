# Copyright 2018 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os

import horovod.tensorflow.keras as hvd
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# to deprecate, use mlrun.mlutils.models and make this model a parameter instead:
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from mlrun import get_or_create_ctx
from mlrun.artifacts import ChartArtifact

# Acquire MLRun context and parameters:
mlctx = get_or_create_ctx("horovod-trainer")
DATA_PATH = mlctx.get_param("data_path")
MODEL_DIR = mlctx.get_param("model_dir", "models")
CHECKPOINTS_DIR = mlctx.get_param("checkpoints_dir")
IMAGE_WIDTH = mlctx.get_param("image_width", 128)
IMAGE_HEIGHT = mlctx.get_param("image_height", 128)
IMAGE_CHANNELS = mlctx.get_param("image_channels", 3)  # RGB color
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
EPOCHS = mlctx.get_param("epochs", 1)
BATCH_SIZE = mlctx.get_param("batch_size", 16)
# RANDOM_STATE must be a parameter for reproducibility:
RANDOM_STATE = mlctx.get_param("random_state", 1)
TEST_SIZE = mlctx.get_param("test_size", 0.2)

# kubeflow outputs/inputs
categories_map = str(mlctx.get_input("categories_map").get())
df = pd.read_csv(str(mlctx.get_input("file_categories")))

# Horovod: initialize Horovod.
hvd.init()

# if gpus found, pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if hvd.rank() == 0:
    mlctx.logger.info(
        f"Validating paths:\nData_path:\t{DATA_PATH}\nModel_dir:\t{MODEL_DIR}\n"
    )
    mlctx.logger.info(f"Categories map:{categories_map}")
    mlctx.logger.info(f"Got {df.shape[0]} files in {DATA_PATH}")
    mlctx.logger.info(f"Training data has {df.size} samples")
    mlctx.logger.info(df.category.value_counts())

# artifact folders (deprecate these)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

#
# Training
#

# Prepare, test, and train the data
train_df, validate_df = train_test_split(
    df, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
train_df["category"] = train_df["category"].astype("str")
validate_df["category"] = validate_df["category"].astype("str")
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

# load model
model = VGG16(include_top=False, input_shape=IMAGE_SHAPE)

# mark loaded layers as not trainable
for layer in model.layers:
    layer.trainable = False

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(128, activation="relu", kernel_initializer="he_uniform")(flat1)
output = Dense(1, activation="sigmoid")(class1)

# define new model
model = Model(inputs=model.inputs, outputs=output)

# Horovod: adjust learning rate based on number of GPUs.
# opt = SGD(lr=0.001, momentum=0.9)
opt = Adadelta(lr=1.0 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(
    loss="binary_crossentropy",
    optimizer=opt,
    experimental_run_tf_function=False,
    metrics=["accuracy"],
)

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
    # Reduce the learning rate if training plateaues, tensorflow.keras callback
    ReduceLROnPlateau(patience=10, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(
        ModelCheckpoint(os.path.join(CHECKPOINTS_DIR, "checkpoint-{epoch}.h5"))
    )

# Set up ImageDataGenerators to do data augmentation for the training images.
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
train_datagen.mean = [123.68, 116.779, 103.939]

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    DATA_PATH,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="binary",
    batch_size=BATCH_SIZE,
)

if hvd.rank() == 0:
    mlctx.logger.info("classes:", train_generator.class_indices)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen.mean = [123.68, 116.779, 103.939]
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    DATA_PATH,
    x_col="filename",
    y_col="category",
    target_size=IMAGE_SIZE,
    class_mode="binary",
    batch_size=BATCH_SIZE,
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=total_train // BATCH_SIZE,
    callbacks=callbacks,
    epochs=EPOCHS,
    verbose=1 if hvd.rank() == 0 else 0,
    validation_data=validation_generator,
    validation_steps=total_validate // BATCH_SIZE,
)

# save the model only on worker 0 to prevent failures ("cannot lock file")
if hvd.rank() == 0:
    # os.makedirs(MODEL_DIR, exist_ok=True)
    model_artifacts = os.path.join(mlctx.artifact_path, MODEL_DIR)

    # log the epoch advancement
    mlctx.logger.info("history:", history.history)
    print("MA:", model_artifacts)

    # Save the model file
    model.save("model.h5")
    # Produce training chart artifact
    chart = ChartArtifact("summary.html")
    chart.header = ["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
    for i in range(EPOCHS):
        chart.add_row(
            [
                i + 1,
                history.history["accuracy"][i],
                history.history["val_accuracy"][i],
                history.history["loss"][i],
                history.history["val_loss"][i],
            ]
        )
    summary = mlctx.log_artifact(
        chart, local_path="training-summary.html", artifact_path=model_artifacts
    )

    # Save weights
    model.save_weights("model-weights.h5")
    weights = mlctx.log_artifact(
        "model-weights",
        local_path="model-weights.h5",
        artifact_path=model_artifacts,
        db_key=False,
    )

    # Log results
    mlctx.log_result("loss", float(history.history["loss"][EPOCHS - 1]))
    mlctx.log_result("accuracy", float(history.history["accuracy"][EPOCHS - 1]))

    mlctx.log_model(
        "model",
        artifact_path=model_artifacts,
        model_file="model.h5",
        labels={"framework": "tensorflow"},
        metrics=mlctx.results,
        extra_data={
            "training-summary": summary,
            "model-architecture.json": bytes(model.to_json(), encoding="utf8"),
            "model-weights.h5": weights,
            "categories_map": mlctx.get_input("categories_map").url,
        },
    )
