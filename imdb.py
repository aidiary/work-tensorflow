from comet_ml import Experiment
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 10


experiment = Experiment(project_name='tf2-imdb',
                        workspace='koichiro-mori',
                        auto_param_logging=False)


def load_dataset():
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_ds, test_ds = dataset['train'], dataset['test']

    # padded_batch()はミニバッチで最長の系列に合わせてpaddingする
    train_ds = train_ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], []))
    test_ds = test_ds.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

    encoder = info.features['text'].encoder

    return train_ds, test_ds, encoder.vocab_size


def main():
    train_ds, test_ds, vocab_size = load_dataset()

    model = Sequential([
        Embedding(vocab_size, 64),
        Bidirectional(LSTM(64, return_sequences=True)),
        # Bidirectionalなので出力ユニット数は64
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    with experiment.train():
        model.fit(train_ds, epochs=NUM_EPOCHS, validation_data=test_ds)

    with experiment.test():
        test_loss, test_acc = model.evaluate(test_ds)
        print('Test Loss: {}'.format(test_loss))
        print('Test Accuracy: {}'.format(test_acc))


if __name__ == "__main__":
    main()
