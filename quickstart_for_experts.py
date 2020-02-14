import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


def load_dataset():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(32)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    return train_ds, test_ds


class MyModel(Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


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
    model = MyModel()
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # training loop
    EPOCHS = 5
    for epoch in range(1, EPOCHS + 1):
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


if __name__ == "__main__":
    main()
