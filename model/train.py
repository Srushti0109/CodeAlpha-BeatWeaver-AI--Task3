from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lstm_model import MusicLSTM
from preprocessor.preprocessor import MIDIPreprocessor


def load_encoded_note_sequence(preprocessor: MIDIPreprocessor) -> tuple[list[int], dict[str, int]]:
    all_events: list[str] = []

    # Reuse MIDI parsing logic from the preprocessor to build one training stream.
    for midi_path in preprocessor._midi_files():
        try:
            all_events.extend(preprocessor._extract_events_from_score(midi_path))
        except Exception as exc:
            print(f"Error processing {midi_path.name}: {exc}")

    vocab_mapping = preprocessor.build_vocabulary()
    encoded_sequence = [vocab_mapping[event] for event in all_events if event in vocab_mapping]
    return encoded_sequence, vocab_mapping


def create_training_data(encoded_sequence: list[int], sequence_length: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if len(encoded_sequence) <= sequence_length:
        raise ValueError(
            f"Not enough notes to build training data. Need > {sequence_length}, got {len(encoded_sequence)}."
        )

    x_data: list[list[int]] = []
    y_data: list[int] = []

    for index in range(len(encoded_sequence) - sequence_length):
        x_data.append(encoded_sequence[index : index + sequence_length])
        y_data.append(encoded_sequence[index + sequence_length])

    x_array = np.array(x_data, dtype=np.int32)
    y_array = np.array(y_data, dtype=np.int32)
    return x_array, y_array


def train() -> None:
    model_output_path = PROJECT_ROOT / "models" / "music_ai_v1.keras"

    preprocessor = MIDIPreprocessor()
    encoded_sequence, vocab_mapping = load_encoded_note_sequence(preprocessor)
    x_train, y_train = create_training_data(encoded_sequence, sequence_length=10)

    music_lstm = MusicLSTM(vocab_size=len(vocab_mapping))
    music_lstm.compile_model()
    music_lstm.model.summary()

    music_lstm.model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=64,
    )

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    music_lstm.model.save(model_output_path)
    print(f"Model saved to: {model_output_path}")


if __name__ == "__main__":
    train()
