from comet_ml import Experiment
import random
import pathlib
import tensorflow as tf
from tensorflow.keras import Model


BATCH_SIZE = 32
NUM_EPOCHS = 10


experiment = Experiment(project_name='tf2-finetune-mobilenet',
                        workspace='koichiro-mori',
                        auto_metric_logging=False)


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0
    # MobileNetの入力は [-1, 1]
    image = 2 * image - 1
    return image


def load_dataset():
    data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                             fname='flower_photos', untar=True)
    data_root = pathlib.Path(data_root_orig)

    # 画像ファイルパスのリストを取得
    all_image_paths = list(data_root.glob('*/*'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    # 画像ファイル数
    image_count = len(all_image_paths)
    print(image_count)

    # ラベル名のリスト（ディレクトリ名）を取得
    label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

    # imageとlabelは同じ順番なのでペアにできる
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    # TODO: ここでtrain/validに分割
    train_ds = image_label_ds

    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=image_count)
    train_ds = train_ds.batch(BATCH_SIZE)
    # prefetchを使うことでモデル訓練中にバックグラウンドでDatasetがbatchを取得できる
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_ds, label_names


class MobileNetV2(Model):

    def __init__(self, output_unit):
        super(MobileNetV2, self).__init__()
        # topは付けずに特徴抽出器として使う
        self.mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
        self.mobile_net.trainable = False

        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(output_unit)

    def call(self, x):
        x = self.mobile_net(x)
        x = self.pooling(x)
        x = self.dense(x)
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
    train_ds, label_names = load_dataset()
    print(train_ds)
    print(label_names)

    model = MobileNetV2(len(label_names))

    # モデルの出力にsoftmaxを使わず確率でないためfrom_logits=Trueにする
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        # reset the metrics
        train_loss.reset_states()
        train_acc.reset_states()

        # training
        for i, (images, labels) in enumerate(train_ds):
            train_step((images, labels), model, criterion, optimizer, (train_loss, train_acc))

        print('Epoch {}, Loss: {}, Accuracy: {}'.format(epoch, train_loss.result(), train_acc.result()))

        experiment.log_metric('train_loss', train_loss.result(), step=epoch)
        experiment.log_metric('train_acc', train_acc.result(), step=epoch)


if __name__ == "__main__":
    main()
