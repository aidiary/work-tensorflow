from comet_ml import Experiment
import os
import random
import pathlib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


BATCH_SIZE = 32
NUM_EPOCHS = 30


experiment = Experiment(project_name='tf2-cats-and-dogs',
                        workspace='koichiro-mori',
                        auto_metric_logging=False)


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150, 150])
    image /= 255.0
    return image


@tf.function
def rotate_tf(image, label):
    random_angles = tf.random.uniform(shape=(tf.shape(image)[0], ),
                                      minval=-45 * np.pi / 180,
                                      maxval=45 * np.pi / 180)
    return tfa.image.rotate(image, random_angles), label


@tf.function
def flip_left_right(image, label):
    return tf.image.random_flip_left_right(image), label


def load_dataset():
    path_to_zip = tf.keras.utils.get_file(
        'cats_and_dogs.zip',
        origin='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip',
        extract=True)
    data_root = pathlib.Path(os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered'))

    train_image_paths = [str(path) for path in list(data_root.glob('train/*/*.jpg'))]
    valid_image_paths = [str(path) for path in list(data_root.glob('validation/*/*.jpg'))]
    random.shuffle(train_image_paths)

    label_to_index = {'cats': 0, 'dogs': 1}

    train_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in train_image_paths]
    valid_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in valid_image_paths]

    # create Dataset
    train_path_ds = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_image_ds = train_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_image_labels, tf.int64))
    train_ds = tf.data.Dataset.zip((train_image_ds, train_label_ds))

    valid_path_ds = tf.data.Dataset.from_tensor_slices(valid_image_paths)
    valid_image_ds = valid_path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(valid_image_labels, tf.int64))
    valid_ds = tf.data.Dataset.zip((valid_image_ds, valid_label_ds))

    num_train_images = len(train_image_paths)
    train_ds = train_ds.cache() \
        .shuffle(buffer_size=num_train_images) \
        .batch(BATCH_SIZE) \
        .map(flip_left_right) \
        .map(rotate_tf) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    valid_ds = valid_ds.cache() \
        .batch(BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, valid_ds


class ConvNet(Model):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(150, 150, 3))
        self.pool1 = MaxPooling2D()
        self.conv2 = Conv2D(32, kernel_size=3, padding='same', activation='relu')
        self.pool2 = MaxPooling2D()
        self.conv3 = Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.pool3 = MaxPooling2D()
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


@tf.function
def train_step(batch, model, criterion, optimizer, metrics):
    images, labels = batch
    train_loss, train_acc = metrics

    with tf.GradientTape() as tape:
        pred = model(images, training=True)
        loss = criterion(labels, pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, pred)


@tf.function
def valid_step(batch, model, criterion, metrics):
    images, labels = batch
    valid_loss, valid_acc = metrics

    pred = model(images, training=False)
    t_loss = criterion(labels, pred)

    valid_loss(t_loss)
    valid_acc(labels, pred)


def main():
    train_ds, valid_ds = load_dataset()
    model = ConvNet()
    criterion = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_acc = tf.keras.metrics.BinaryAccuracy(name='valid_accuracy')

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # reset the metrics
        train_loss.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_acc.reset_states()

        # training
        for i, (images, labels) in enumerate(train_ds):
            train_step((images, labels), model, criterion, optimizer, (train_loss, train_acc))

        # validation
        for i, (images, labels) in enumerate(valid_ds):
            valid_step((images, labels), model, criterion, (valid_loss, valid_acc))

        print('Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'.format(
            epoch, train_loss.result(), train_acc.result(),
            valid_loss.result(), valid_acc.result()))

        experiment.log_metric('train_loss', train_loss.result(), step=epoch)
        experiment.log_metric('train_acc', train_acc.result(), step=epoch)
        experiment.log_metric('valid_loss', valid_loss.result(), step=epoch)
        experiment.log_metric('valid_acc', valid_acc.result(), step=epoch)


if __name__ == "__main__":
    main()
