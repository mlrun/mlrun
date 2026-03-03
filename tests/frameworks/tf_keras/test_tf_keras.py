# Copyright 2023 Iguazio
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

import json
import tempfile

import numpy as np
import pytest

try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    import mlrun
    from mlrun.frameworks.tf_keras import TFKerasModelHandler, apply_mlrun
    from mlrun.frameworks.tf_keras.utils import is_keras_3
except ImportError:
    # just so pytest doesn't fail
    def is_keras_3():
        return False


def preprocess_data(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Normalize pixel values to be between 0 and 1
    x = x.astype("float32") / 255.0

    # Reshape images to (num_samples, 28, 28, 1) for CNN input
    x = np.expand_dims(x, -1)

    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes=10)

    return x, y


def train(epochs=1, batch_size=32):
    # Load the MNIST dataset
    mlrun.utils.logger.info("\nLoading MNIST dataset...")
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
    x_train, y_train = preprocess_data(x=x_train, y=y_train)
    x_val, y_val = preprocess_data(x=x_val, y=y_val)
    training_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
        batch_size=batch_size
    )
    validation_set = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(
        batch_size=batch_size
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    training_set = training_set.with_options(options)
    validation_set = validation_set.with_options(options)

    print("Building a small Keras model...")
    # Define a simple convolutional neural network (CNN) model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    # Use Adam optimizer and categorical crossentropy for multi-class classification
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    print("\nApplying MLRun to the model...")
    apply_mlrun(model=model, model_name="mnist_cpu_model")

    # Train the model
    print(
        f"\nStarting training for {epochs} epochs with batch size {batch_size} on CPU..."
    )
    model.fit(
        training_set,
        epochs=epochs,
        validation_data=validation_set,
        verbose=0,
    )

    return model


def evaluate(model: "tf.keras.Model") -> dict:
    # Load the model:
    model_path = None
    if not isinstance(model, "tf.keras.Model"):
        model_path = model
        model_handler = TFKerasModelHandler(model_path=model)
        model_handler.load()
        model = model_handler.model

    # Load the MNIST dataset
    mlrun.utils.logger.info("\nLoading MNIST dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    x_test, y_test = preprocess_data(x=x_test, y=y_test)

    mlrun.utils.logger.info("\nApplying MLRun to the model...")
    apply_mlrun(model=model, model_path=model_path, model_name="mnist_cpu_model")

    # Evaluate the model on the test data
    mlrun.utils.logger.info("\nEvaluating the model on the test set...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    mlrun.utils.logger.info(f"Test Loss: {loss:.4f}")
    mlrun.utils.logger.info(f"Test Accuracy: {accuracy:.4f}")

    return {
        "loss": loss,
        "accuracy": accuracy,
    }


@pytest.mark.skipif(
    is_keras_3(),
    reason="Keras 3 test is freezing due to dead lock issue when running within PyTest.",
)
def test_training(rundb_mock):
    """
    Test the `apply_mlrun` function with a simple TensorFlow Keras model training.
    """
    with tempfile.TemporaryDirectory() as test_directory:
        # Run training:
        train_run = mlrun.new_function().run(
            artifact_path=test_directory,
            handler=train,
            local=True,
        )

        # Print the outputs for manual validation:
        mlrun.utils.logger.info(json.dumps(train_run.outputs, indent=4))

        # Get assertion parameters:
        expected_artifacts = [
            "accuracy_summary.html",
            "learning_rate_values.html",
            "loss_summary.html",
            "model",
            "training_accuracy.html",
            "training_loss.html",
            "validation_accuracy.html",
            "validation_loss.html",
        ]
        expected_results = [
            "learning_rate",
            "training_accuracy",
            "training_loss",
            "validation_accuracy",
            "validation_loss",
        ]

        # Validate artifacts:
        for expected_artifact in expected_artifacts:
            assert expected_artifact in train_run.status.artifact_uris
        assert len(train_run.status.artifacts) == len(expected_artifacts)

        # Validate results:
        for expected_result in expected_results:
            assert expected_result in train_run.status.results
        assert (
            len(train_run.status.results) == len(expected_results) + 1
        )  # +1 for the returned model.


@pytest.mark.skipif(
    is_keras_3(),
    reason="Keras 3 test is freezing due to dead lock issue when running within PyTest.",
)
def test_evaluation(rundb_mock):
    """
    Test the `apply_mlrun` function with a simple TensorFlow Keras model training to evaluation flow.
    """
    with tempfile.TemporaryDirectory() as test_directory:
        # Run training:
        train_run = mlrun.new_function().run(
            artifact_path=test_directory,
            handler=train,
            local=True,
        )

        # Run evaluation (on the model that was just trained):
        evaluate_run = mlrun.new_function().run(
            artifact_path=test_directory,
            handler=evaluate,
            params={
                "model": train_run.outputs["model"],
            },
            local=True,
        )

        # Print the outputs for manual validation:
        mlrun.utils.logger.info(json.dumps(evaluate_run.outputs, indent=4))

        # Get assertion parameters:
        expected_artifacts = ["evaluation_loss.html", "evaluation_accuracy.html"]
        expected_results = ["evaluation_loss", "evaluation_accuracy"]

        # Validate artifacts:
        for expected_artifact in expected_artifacts:
            assert expected_artifact in evaluate_run.status.artifact_uris
        assert len(evaluate_run.status.artifacts) == len(expected_artifacts)

        # Validate results:
        for expected_result in expected_results:
            assert expected_result in evaluate_run.status.results
        assert (
            len(evaluate_run.status.results) == len(expected_results) + 2
        )  # +1 for the returned dictionary and +1 for the updated model.
