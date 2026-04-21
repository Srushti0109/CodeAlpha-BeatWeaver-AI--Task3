# Requirements Document

## Introduction

This document defines the requirements for the AI Music Generation feature. The system collects MIDI files, preprocesses them using music21, trains an LSTM model with TensorFlow/Keras to learn musical patterns, and exposes a FastAPI backend for on-demand music generation. Generated compositions and user data are persisted in Supabase, enabling a scalable, cloud-backed music generation service.

The system is composed of four major subsystems: a MIDI collection pipeline, a music21-based preprocessing pipeline, an LSTM model training and inference engine, and a FastAPI service layer backed by Supabase for storage and user management.

## Glossary

- **MIDI_Collector**: The component responsible for discovering, downloading, and validating raw MIDI files from configured sources.
- **Preprocessor**: The music21-based component that parses MIDI files, extracts musical elements, and encodes them into integer sequences for LSTM training.
- **LSTM_Model**: The TensorFlow/Keras LSTM model that learns musical patterns and generates new note sequences.
- **MIDI_Renderer**: The component that converts decoded note/chord token sequences back into MIDI bytes.
- **API**: The FastAPI backend that exposes REST endpoints for music generation and track management.
- **Storage**: The Supabase Storage service used to persist MIDI and audio files.
- **Database**: The Supabase PostgreSQL database used to persist track metadata and user records.
- **Token**: A string representation of a musical element — either a note name (e.g. `"C4"`), a chord string (e.g. `"C4.E4.G4"`), or `"rest"`.
- **VocabularyMapping**: A bijective mapping between string tokens and integer indices, used to encode and decode musical sequences.
- **GenerationRequest**: The request payload sent by a client to trigger music generation.
- **TrackRecord**: The metadata record stored in the Database for a generated music track.
- **Seed**: An optional list of tokens provided by the client to guide the start of a generated sequence.
- **Temperature**: A float parameter controlling the randomness of generation; lower values produce more deterministic output.

---

## Requirements

### Requirement 1: MIDI Collection

**User Story:** As a system operator, I want to collect MIDI files from local and remote sources, so that I can build a training corpus for the LSTM model.

#### Acceptance Criteria

1. WHEN a list of source paths or URLs is provided, THE MIDI_Collector SHALL traverse each source and return a list of valid MIDI file paths.
2. WHEN a file path is provided for validation, THE MIDI_Collector SHALL return true if and only if the file is parseable by music21.
3. WHEN a list of MIDI file paths is provided for deduplication, THE MIDI_Collector SHALL return a list containing only unique files, determined by content hash.
4. WHEN collection completes, THE MIDI_Collector SHALL emit a manifest file listing all collected file paths.
5. IF a remote MIDI file cannot be downloaded, THEN THE MIDI_Collector SHALL log the error and continue collecting remaining sources.

---

### Requirement 2: Music21 Preprocessing

**User Story:** As a data engineer, I want to preprocess MIDI files into encoded integer sequences, so that the LSTM model can be trained on structured musical data.

#### Acceptance Criteria

1. WHEN a valid MIDI file path is provided, THE Preprocessor SHALL parse it into a music21 Score object.
2. WHEN a music21 Score object is provided, THE Preprocessor SHALL extract all notes, chords, and rests as an ordered list of string tokens preserving temporal order.
3. WHEN a list of token lists is provided, THE Preprocessor SHALL build a VocabularyMapping where each unique token maps to a unique integer and the mapping is bijective.
4. WHEN a token list, a VocabularyMapping, and a sequence length are provided, THE Preprocessor SHALL produce sliding-window (X, y) training arrays where X has shape (N, sequence_length) and N equals the number of elements minus the sequence length.
5. WHEN a list of integer indices and a VocabularyMapping are provided, THE Preprocessor SHALL decode the indices back into the original token strings.
6. THE Preprocessor SHALL normalize note durations to quantized values during extraction.
7. IF a MIDI file cannot be parsed by music21, THEN THE Preprocessor SHALL log the error with the file path, skip the file, and continue processing remaining files.
8. IF preprocessing yields fewer than sequence_length plus one total elements, THEN THE Preprocessor SHALL raise a descriptive error before training begins.

---

### Requirement 3: Vocabulary Round-Trip

**User Story:** As a developer, I want the vocabulary encoding and decoding to be lossless, so that musical information is preserved through the preprocessing pipeline.

#### Acceptance Criteria

1. THE Preprocessor SHALL produce a VocabularyMapping such that for any token in the vocabulary, encoding then decoding returns the original token.
2. THE VocabularyMapping SHALL expose a `vocab_size` equal to the number of unique tokens in the source corpus.
3. THE Preprocessor SHALL persist the VocabularyMapping to disk alongside the trained model weights.

---

### Requirement 4: LSTM Model Training

**User Story:** As a data scientist, I want to train an LSTM model on encoded musical sequences, so that the model learns to generate coherent music.

#### Acceptance Criteria

1. WHEN vocabulary size and sequence length are provided, THE LSTM_Model SHALL construct and compile a model with an embedding layer, stacked LSTM layers, and a Dense softmax output layer.
2. WHEN training data arrays and hyperparameters are provided, THE LSTM_Model SHALL train the model and save checkpoints to the specified directory.
3. WHILE training is in progress, THE LSTM_Model SHALL apply ModelCheckpoint and EarlyStopping callbacks.
4. WHEN training completes, THE LSTM_Model SHALL serialize the model weights and VocabularyMapping to disk.
5. WHEN saved weights and vocabulary paths are provided, THE LSTM_Model SHALL load them and make the model ready for inference.

---

### Requirement 5: Music Generation

**User Story:** As a developer, I want to generate new musical sequences from a trained LSTM model, so that the system can produce original compositions on demand.

#### Acceptance Criteria

1. WHEN a seed sequence, length, and temperature are provided, THE LSTM_Model SHALL generate a list of exactly `length` integer indices.
2. THE LSTM_Model SHALL ensure all generated indices are in the range [0, vocab_size).
3. WHEN generating a sequence, THE LSTM_Model SHALL apply temperature-scaled softmax sampling at each step.
4. WHEN a lower temperature is used compared to a higher temperature, THE LSTM_Model SHALL produce output with lower entropy (more deterministic distribution) over a sufficiently long sequence.
5. IF the model weights are not loaded when generation is requested, THEN THE LSTM_Model SHALL raise an error indicating the model is not ready.

---

### Requirement 6: MIDI Rendering

**User Story:** As a developer, I want to convert generated token sequences into MIDI files, so that users can download and play the generated music.

#### Acceptance Criteria

1. WHEN a list of string tokens and a tempo value are provided, THE MIDI_Renderer SHALL produce valid MIDI bytes containing exactly one note, chord, or rest event per token.
2. WHEN a token is the string `"rest"`, THE MIDI_Renderer SHALL insert a rest event at the corresponding position.
3. WHEN a token contains a `"."` separator, THE MIDI_Renderer SHALL interpret it as a chord and insert all specified pitches simultaneously.
4. WHEN a token is a plain note name, THE MIDI_Renderer SHALL insert a single note event at the corresponding position.
5. THE MIDI_Renderer SHALL produce MIDI bytes that are re-parseable by music21 without error.
6. WHERE audio rendering is enabled, THE MIDI_Renderer SHALL convert MIDI bytes to WAV audio using FluidSynth.

---

### Requirement 7: MIDI Rendering Round-Trip

**User Story:** As a developer, I want rendered MIDI to be re-parseable, so that the output is a valid, standards-compliant MIDI file.

#### Acceptance Criteria

1. FOR ALL valid token sequences, rendering to MIDI bytes and then re-parsing with music21 SHALL produce a non-null Score object.
2. THE MIDI_Renderer SHALL embed the specified tempo as a MetronomeMark in the output MIDI stream.

---

### Requirement 8: FastAPI Generation Endpoint

**User Story:** As a client application, I want to trigger music generation via a REST API, so that I can request new compositions without managing the model directly.

#### Acceptance Criteria

1. WHEN a POST request is made to `/generate` with a valid GenerationRequest body, THE API SHALL invoke the LSTM_Model to generate a sequence, render it to MIDI, upload it to Storage, insert a TrackRecord into the Database, and return a response containing `track_id`, `download_url`, and metadata.
2. WHEN a GenerationRequest contains `seed_tokens`, THE API SHALL use those tokens as the seed for generation.
3. IF the model is not loaded when a POST request is made to `/generate`, THEN THE API SHALL return HTTP 503 with `{"error": "Model not ready"}`.
4. IF the Supabase Storage upload fails after three retry attempts with exponential backoff, THEN THE API SHALL return HTTP 502 with `{"error": "Storage unavailable"}`.
5. IF a GenerationRequest contains a `temperature` or `length` value outside the allowed bounds, THEN THE API SHALL return HTTP 422 with field-level validation error details.

---

### Requirement 9: GenerationRequest Validation

**User Story:** As a client application, I want the API to validate generation parameters, so that invalid requests are rejected before reaching the model.

#### Acceptance Criteria

1. THE API SHALL reject any GenerationRequest where `length` is less than 16 or greater than 512.
2. THE API SHALL reject any GenerationRequest where `temperature` is less than 0.1 or greater than 2.0.
3. THE API SHALL reject any GenerationRequest where `tempo` is less than 40 or greater than 240.
4. WHEN a GenerationRequest passes all validation rules, THE API SHALL forward it to the LSTM_Model for generation.

---

### Requirement 10: Track Management Endpoints

**User Story:** As a user, I want to list, retrieve, and delete my generated tracks, so that I can manage my music library.

#### Acceptance Criteria

1. WHEN a GET request is made to `/tracks` with a valid `user_id`, THE API SHALL return a list of all TrackRecord objects belonging to that user.
2. WHEN a GET request is made to `/tracks/{track_id}` with a valid `track_id`, THE API SHALL return the corresponding TrackRecord including a signed download URL.
3. WHEN a DELETE request is made to `/tracks/{track_id}`, THE API SHALL remove the TrackRecord from the Database and delete the associated file from Storage.
4. WHEN a GET request is made to `/health`, THE API SHALL return a response indicating the service status.

---

### Requirement 11: Supabase Storage Integration

**User Story:** As a system operator, I want generated MIDI files to be stored in Supabase Storage, so that users can download their tracks via signed URLs.

#### Acceptance Criteria

1. WHEN MIDI bytes and a user ID are provided, THE Storage SHALL store the file under the path `users/{user_id}/tracks/` and return a signed URL.
2. THE Storage SHALL enforce per-user path isolation so that one user cannot access another user's files.
3. WHEN a track is deleted, THE Storage SHALL remove the associated storage object.
4. THE Storage SHALL generate signed URLs with a time-to-live of one hour.

---

### Requirement 12: Supabase Database Integration

**User Story:** As a system operator, I want track metadata to be persisted in a PostgreSQL database, so that users can retrieve their generation history.

#### Acceptance Criteria

1. WHEN a TrackRecord is provided, THE Database SHALL insert it and return the generated `track_id`.
2. WHEN a `user_id` is provided, THE Database SHALL return all TrackRecord objects associated with that user.
3. WHEN a `track_id` and `user_id` are provided for deletion, THE Database SHALL remove the corresponding TrackRecord.

---

### Requirement 13: Authentication and Security

**User Story:** As a system operator, I want all API endpoints to be protected by authentication, so that only authorized users can generate and access tracks.

#### Acceptance Criteria

1. WHEN a request is made to any API endpoint, THE API SHALL validate the Supabase JWT token provided in the `Authorization: Bearer <token>` header.
2. IF a request is made without a valid JWT token, THEN THE API SHALL return HTTP 401.
3. THE API SHALL load the Supabase URL and service role key from environment variables and never from hardcoded values.
4. WHILE a user is authenticated, THE API SHALL enforce a rate limit of 10 POST requests per minute to `/generate` per user.
5. IF a user exceeds the rate limit, THEN THE API SHALL return HTTP 429.
