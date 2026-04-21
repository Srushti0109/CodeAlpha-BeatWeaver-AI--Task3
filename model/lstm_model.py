from __future__ import annotations

import tensorflow as tf


class MusicLSTM:
    def __init__(
        self,
        vocab_size: int = 53,
        embedding_dim: int = 128,
        lstm_units: int = 256,
        dropout_rate: float = 0.3,
    ) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=self.vocab_size,
                    output_dim=self.embedding_dim,
                ),
                tf.keras.layers.LSTM(self.lstm_units, return_sequences=True),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.LSTM(self.lstm_units),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.vocab_size, activation="softmax"),
            ]
        )
        return model

    def compile_model(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )


if __name__ == "__main__":
    music_lstm = MusicLSTM(vocab_size=53)
    music_lstm.compile_model()
    music_lstm.model.build(input_shape=(None, None))
    music_lstm.model.summary()
