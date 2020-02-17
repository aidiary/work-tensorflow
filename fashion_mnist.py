from comet_ml import Experiment
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import Model


config = {
    'num_epochs': 5,
    'batch_size': 32
}

experiment = Experiment(project_name='tf2-fashion-mnist',
                        workspace='koichiro-mori',
                        auto_metric_logging=False)
experiment.log_parameters(config)


def load_dataset():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # scaling
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = tf.cast(train_images, tf.float32)
    test_images = tf.cast(test_images, tf.float32)

    # create dataset
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)).shuffle(len(train_images)).batch(config['batch_size'])

    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)).batch(config['batch_size'])

    return train_ds, test_ds


class MultiLayerPerceptron(Model):

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
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
def test_step(batch, model, criterion, metrics):
    images, labels = batch
    test_loss, test_acc = metrics

    pred = model(images, training=False)
    t_loss = criterion(labels, pred)

    test_loss(t_loss)
    test_acc(labels, pred)


def main():
    train_ds, test_ds = load_dataset()
    model = MultiLayerPerceptron()
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # training loop
    for epoch in range(1, config['num_epochs'] + 1):
        # reset the metrics
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        # training
        for images, labels in train_ds:
            train_step((images, labels),
                       model,
                       criterion,
                       optimizer,
                       (train_loss, train_acc))

        # validation
        for test_images, test_labels in test_ds:
            test_step((test_images, test_labels),
                      model,
                      criterion,
                      (test_loss, test_acc))

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch,
                              train_loss.result(),
                              train_acc.result(),
                              test_loss.result(),
                              test_acc.result()))

        # logging to Comet.ML
        experiment.log_metric('train_loss', train_loss.result(), step=epoch)
        experiment.log_metric('train_acc', train_acc.result(), step=epoch)
        experiment.log_metric('test_loss', test_loss.result(), step=epoch)
        experiment.log_metric('test_acc', test_acc.result(), step=epoch)


if __name__ == "__main__":
    main()
