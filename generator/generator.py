from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from music21 import chord, note, stream

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessor.preprocessor import MIDIPreprocessor


def load_vocab(vocab_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    with vocab_path.open("r", encoding="utf-8") as file:
        vocab_mapping: dict[str, int] = json.load(file)
    reverse_vocab = {index: token for token, index in vocab_mapping.items()}
    return vocab_mapping, reverse_vocab


def load_encoded_note_sequence(preprocessor: MIDIPreprocessor, vocab_mapping: dict[str, int]) -> list[int]:
    all_events: list[str] = []

    for midi_path in preprocessor._midi_files():
        try:
            all_events.extend(preprocessor._extract_events_from_score(midi_path))
        except Exception as exc:
            print(f"Error processing {midi_path.name}: {exc}")

    return [vocab_mapping[event] for event in all_events if event in vocab_mapping]


def token_to_music21_object(token: str):
    if "." in token:
        chord_pitches = [int(value) + 60 for value in token.split(".")]
        return chord.Chord(chord_pitches)
    return note.Note(token)


def generate_music(
    model: tf.keras.Model | None = None,
    vocab_mapping: dict[str, int] | None = None,
) -> Path:
    model_path = PROJECT_ROOT / "models" / "music_ai_v1.keras"
    vocab_path = PROJECT_ROOT / "models" / "vocab.json"
    output_path = PROJECT_ROOT / "output" / "generated_lofi.mid"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    if vocab_mapping is None:
        vocab_mapping, reverse_vocab = load_vocab(vocab_path)
    else:
        reverse_vocab = {index: token for token, index in vocab_mapping.items()}

    preprocessor = MIDIPreprocessor()
    encoded_sequence = load_encoded_note_sequence(preprocessor, vocab_mapping)

    sequence_length = 10
    if len(encoded_sequence) <= sequence_length:
        raise ValueError("Not enough encoded notes to create a seed sequence of length 10.")

    if model is None:
        model = tf.keras.models.load_model(model_path)

    start_index = random.randint(0, len(encoded_sequence) - sequence_length - 1)
    seed = encoded_sequence[start_index : start_index + sequence_length]
    generated_indices: list[int] = []

    for _ in range(100):
        model_input = np.array([seed], dtype=np.int32)
        predictions = model.predict(model_input, verbose=0)[0]
        next_index = int(np.argmax(predictions))
        generated_indices.append(next_index)
        seed = seed[1:] + [next_index]

    generated_stream = stream.Stream()
    current_offset = 0.0

    for index in generated_indices:
        token = reverse_vocab[index]
        music_obj = token_to_music21_object(token)
        music_obj.offset = current_offset
        generated_stream.append(music_obj)
        current_offset += 0.5

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_stream.write("midi", fp=str(output_path))
    print(f"Generated MIDI saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_music()
