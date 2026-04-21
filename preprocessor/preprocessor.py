from __future__ import annotations

import json
from pathlib import Path

from music21 import chord, converter, instrument, note


class MIDIPreprocessor:
    def __init__(self, data_dir: Path | None = None, output_path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.data_dir = data_dir or (project_root / "data")
        self.output_path = output_path or (project_root / "models" / "vocab.json")

    def _midi_files(self) -> list[Path]:
        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data folder not found: {self.data_dir}")

        files = sorted(
            [*self.data_dir.glob("*.mid"), *self.data_dir.glob("*.midi")],
            key=lambda path: path.name.lower(),
        )
        if not files:
            raise FileNotFoundError(f"No MIDI files found in: {self.data_dir}")
        return files

    def _extract_events_from_score(self, midi_path: Path) -> list[str]:
        score = converter.parse(midi_path)
        events: list[str] = []

        parsed_parts = instrument.partitionByInstrument(score)
        stream_to_read = parsed_parts.parts if parsed_parts is not None else [score.flat]

        for part in stream_to_read:
            for element in part.recurse():
                if isinstance(element, note.Note):
                    events.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    chord_token = ".".join(str(pitch) for pitch in element.normalOrder)
                    events.append(chord_token)

        return events

    def build_vocabulary(self) -> dict[str, int]:
        all_events: list[str] = []

        for midi_path in self._midi_files():
            try:
                all_events.extend(self._extract_events_from_score(midi_path))
            except Exception as exc:
                print(f"Error processing {midi_path.name}: {exc}")

        unique_events = sorted(set(all_events))
        vocab_mapping = {event: index for index, event in enumerate(unique_events)}
        return vocab_mapping

    def save_vocabulary(self, vocab_mapping: dict[str, int]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(vocab_mapping, file, indent=2, sort_keys=True)

    def run(self) -> None:
        vocab_mapping = self.build_vocabulary()
        self.save_vocabulary(vocab_mapping)
        print(f"Vocab Size: {len(vocab_mapping)}")


if __name__ == "__main__":
    MIDIPreprocessor().run()
