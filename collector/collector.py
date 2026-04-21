from __future__ import annotations

from pathlib import Path

from music21 import converter


def collect_midi_files() -> None:
    # data/ is one level up from collector/
    data_dir = Path(__file__).resolve().parent.parent / "data"

    if not data_dir.exists() or not data_dir.is_dir():
        print(f"Data folder not found: {data_dir}")
        return

    midi_files = sorted(
        [*data_dir.glob("*.mid"), *data_dir.glob("*.midi")],
        key=lambda path: path.name.lower(),
    )

    if not midi_files:
        print(f"No MIDI files found in: {data_dir}")
        return

    for midi_path in midi_files:
        try:
            score = converter.parse(midi_path)
            part_count = len(score.parts)
            print(f"Successfully loaded: {midi_path.name}")
            print(f"Parts/Instruments found: {part_count}")
        except Exception as exc:
            print(f"Error loading {midi_path.name}: {exc}")


if __name__ == "__main__":
    collect_midi_files()
