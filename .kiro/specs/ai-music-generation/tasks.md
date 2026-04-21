# Implementation Plan: AI Music Generation

## Overview

Implement an end-to-end AI music generation system in Python: a MIDI collection pipeline, a music21-based preprocessing pipeline, an LSTM model (TensorFlow/Keras), a MIDI renderer, and a FastAPI backend backed by Supabase for storage and user management.

## Tasks

- [ ] 1. Set up project structure, dependencies, and core data models
  - Create directory layout: `collector/`, `preprocessor/`, `model/`, `renderer/`, `api/`, `tests/`
  - Create `requirements.txt` with all pinned dependencies (music21, tensorflow, numpy, fastapi, uvicorn, pydantic, supabase-py, httpx, hypothesis, pytest, python-dotenv)
  - Implement `GenerationRequest`, `TrackRecord`, and `VocabularyMapping` Pydantic models in `api/models.py`
  - Add `.env.example` with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` placeholders
  - _Requirements: 9.1, 9.2, 9.3, 11.1, 12.1_

- [ ] 2. Implement MIDI Collection Pipeline
  - [ ] 2.1 Implement `MIDICollector` class in `collector/collector.py`
    - Implement `collect(sources)` to traverse local directories and download remote `.mid` files
    - Implement `validate(path)` using `music21.converter.parse` to check parseability
    - Implement `deduplicate(paths)` using SHA256 content hashing
    - Emit a JSON manifest file listing all collected paths on completion
    - Log and skip remote files that cannot be downloaded
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test for deduplication idempotence
    - **Property 13: Deduplication Idempotence**
    - **Validates: Requirements 1.3**
    - Use `hypothesis` to generate lists of paths with duplicates; assert no two entries share a content hash and re-running deduplication returns the same list

  - [ ]* 2.3 Write unit tests for `MIDICollector`
    - Test `validate` returns `True` for a known-good fixture and `False` for a corrupt file
    - Test `deduplicate` removes exact-content duplicates
    - Test manifest file is written with correct paths
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 3. Implement music21 Preprocessing Pipeline
  - [ ] 3.1 Implement `MusicPreprocessor` class in `preprocessor/preprocessor.py`
    - Implement `parse_midi(path)` returning a `music21.stream.Score`
    - Implement `extract_elements(score)` extracting notes, chords, and rests as ordered string tokens with quantized durations
    - Implement `build_vocabulary(all_elements)` returning a bijective `VocabularyMapping`
    - Implement `encode_sequences(elements, vocab, sequence_length)` producing sliding-window `(X, y)` arrays
    - Implement `decode_sequence(indices, vocab)` converting integer indices back to token strings
    - Persist `VocabularyMapping` to disk as JSON alongside model weights
    - Log and skip unparseable files; raise `ValueError` if total elements < `sequence_length + 1`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3_

  - [ ]* 3.2 Write property test for vocabulary round-trip
    - **Property 1: Vocabulary Round-Trip**
    - **Validates: Requirements 3.1, 2.3, 2.5**
    - Use `hypothesis` to generate arbitrary token lists; assert `decode(encode(token)) == token` for every token

  - [ ]* 3.3 Write property test for vocabulary size invariant
    - **Property 2: Vocabulary Size Invariant**
    - **Validates: Requirements 3.2, 2.3**
    - Use `hypothesis` to generate token lists; assert `vocab.vocab_size == len(set(all_tokens))`

  - [ ]* 3.4 Write property test for sliding window shape
    - **Property 3: Sliding Window Shape**
    - **Validates: Requirements 2.4**
    - Use `hypothesis` to generate token lists of length L and sequence length S (L > S); assert `len(X) == L - S`

  - [ ]* 3.5 Write unit tests for `MusicPreprocessor`
    - Test `extract_elements` against a known MIDI fixture with expected token output
    - Test `build_vocabulary` for bijectivity
    - Test `encode_sequences` produces correct `(N, sequence_length)` shape
    - Test `decode_sequence` recovers original tokens
    - Test error is raised when corpus is too small
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.8_

- [ ] 4. Checkpoint — Ensure all preprocessing tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement LSTM Model
  - [ ] 5.1 Implement `MusicLSTM` class in `model/lstm.py`
    - Implement `build(vocab_size, sequence_length, embedding_dim, lstm_units, dropout)` constructing and compiling the Keras model (Embedding → stacked LSTM → Dense softmax)
    - Implement `train(X, y, epochs, batch_size, checkpoint_dir)` with `ModelCheckpoint` and `EarlyStopping` callbacks
    - Implement `generate(seed_sequence, length, temperature)` with temperature-scaled softmax sampling and sliding-window inference
    - Implement `load(weights_path, vocab_path)` to restore model and vocabulary for inference
    - Serialize model weights and `VocabularyMapping` to disk after training
    - Raise an error if `generate` is called before weights are loaded
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 5.2 Write property test for generation length
    - **Property 4: Generation Length**
    - **Validates: Requirements 5.1**
    - Use `hypothesis` to generate valid `length` (16–512) and `temperature` (0.1–2.0) values; assert `len(result) == length`

  - [ ]* 5.3 Write property test for valid vocabulary indices
    - **Property 5: Valid Vocabulary Indices**
    - **Validates: Requirements 5.2**
    - Use `hypothesis` with varied seeds and lengths; assert all returned indices are in `[0, vocab_size)`

  - [ ]* 5.4 Write property test for temperature entropy monotonicity
    - **Property 6: Temperature Entropy Monotonicity**
    - **Validates: Requirements 5.4**
    - Generate sequences of ≥ 512 tokens at low vs. high temperature; assert entropy at low temperature < entropy at high temperature

  - [ ]* 5.5 Write unit tests for `MusicLSTM`
    - Test model builds without error for various vocab sizes
    - Test `generate` raises error when model not loaded
    - Test temperature scaling shifts probability distributions correctly
    - _Requirements: 4.1, 5.3, 5.5_

- [ ] 6. Implement MIDI Renderer
  - [ ] 6.1 Implement `MIDIRenderer` class in `renderer/renderer.py`
    - Implement `sequence_to_midi(tokens, tempo, instrument)` converting token list to MIDI bytes using music21
    - Handle `"rest"` tokens as rest events, `"."` -separated tokens as chords, plain tokens as single notes
    - Embed `MetronomeMark` matching the specified tempo
    - Implement optional `midi_to_audio(midi_bytes)` using FluidSynth for WAV rendering
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 7.1, 7.2_

  - [ ]* 6.2 Write property test for MIDI round-trip
    - **Property 7: MIDI Round-Trip**
    - **Validates: Requirements 6.5, 7.1**
    - Use `hypothesis` to generate non-empty valid token lists and tempos in [40, 240]; assert re-parsing the rendered MIDI bytes with music21 returns a non-null Score without error

  - [ ]* 6.3 Write property test for MIDI event count
    - **Property 8: MIDI Event Count**
    - **Validates: Requirements 6.1**
    - Use `hypothesis` to generate token lists; assert rendered MIDI contains exactly one note/chord/rest event per token

  - [ ]* 6.4 Write property test for MIDI tempo embedding
    - **Property 9: MIDI Tempo Embedding**
    - **Validates: Requirements 7.2**
    - Use `hypothesis` to generate tempo values in [40, 240]; assert rendered MIDI contains a `MetronomeMark` matching the specified tempo

  - [ ]* 6.5 Write unit tests for `MIDIRenderer`
    - Test rest, chord, and note token types produce correct event types
    - Test round-trip: render → re-parse → verify note count matches token count
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 7. Checkpoint — Ensure all model and renderer tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement Supabase Integration
  - [ ] 8.1 Implement `SupabaseClient` class in `api/supabase_client.py`
    - Implement `upload_midi(midi_bytes, user_id)` storing files under `users/{user_id}/tracks/` and returning a signed URL with 1-hour TTL
    - Implement `insert_track(record)` inserting a `TrackRecord` and returning the generated `track_id`
    - Implement `get_tracks(user_id)` returning all `TrackRecord` objects for a user
    - Implement `delete_track(track_id, user_id)` removing the DB record and storage object
    - Load `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` from environment variables only
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 12.1, 12.2, 12.3, 13.3_

  - [ ]* 8.2 Write property test for track listing isolation
    - **Property 11: Track Listing Isolation**
    - **Validates: Requirements 10.1, 12.2**
    - Use `hypothesis` to generate distinct user IDs; assert all returned `TrackRecord` objects have `user_id` matching the requested user

  - [ ]* 8.3 Write property test for storage path isolation
    - **Property 12: Storage Path Isolation**
    - **Validates: Requirements 11.2**
    - Use `hypothesis` to generate two distinct user IDs; assert their storage paths share no common prefix beyond the bucket root

  - [ ]* 8.4 Write unit tests for `SupabaseClient`
    - Test upload stores file at correct path and returns a signed URL
    - Test `get_tracks` returns only records for the requested user
    - Test `delete_track` removes both DB record and storage object
    - _Requirements: 11.1, 11.2, 11.3, 12.1, 12.2, 12.3_

- [ ] 9. Implement FastAPI Backend
  - [ ] 9.1 Implement JWT authentication middleware in `api/auth.py`
    - Validate `Authorization: Bearer <token>` header on every request using Supabase JWT verification
    - Return HTTP 401 for missing or invalid tokens
    - _Requirements: 13.1, 13.2_

  - [ ] 9.2 Implement per-user rate limiting in `api/rate_limit.py`
    - Track POST `/generate` request counts per user within a 60-second sliding window
    - Return HTTP 429 when a user exceeds 10 requests per minute
    - _Requirements: 13.4, 13.5_

  - [ ] 9.3 Implement `POST /generate` endpoint in `api/routes/generate.py`
    - Validate `GenerationRequest` via Pydantic (length 16–512, temperature 0.1–2.0, tempo 40–240)
    - Invoke `MusicLSTM.generate`, decode via `MusicPreprocessor.decode_sequence`, render via `MIDIRenderer.sequence_to_midi`
    - Upload MIDI to Supabase Storage with exponential backoff (max 3 retries); return HTTP 502 on failure
    - Insert `TrackRecord` into Supabase DB; return `{track_id, download_url, metadata}`
    - Return HTTP 503 if model is not loaded
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 9.2, 9.3, 9.4_

  - [ ] 9.4 Implement track management endpoints in `api/routes/tracks.py`
    - `GET /tracks` — return all `TrackRecord` objects for the authenticated user
    - `GET /tracks/{track_id}` — return a single `TrackRecord` with a fresh signed URL
    - `DELETE /tracks/{track_id}` — delete DB record and storage object
    - `GET /health` — return service status
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ] 9.5 Wire all components in `api/main.py`
    - Load model weights and vocabulary at startup via FastAPI lifespan event
    - Register auth middleware, rate limiter, and all routers
    - Cache loaded `MusicLSTM` instance in app state
    - _Requirements: 8.1, 8.3, 13.1_

  - [ ]* 9.6 Write property test for GenerationRequest validation
    - **Property 10: GenerationRequest Validation Rejects Out-of-Bounds Inputs**
    - **Validates: Requirements 8.5, 9.1, 9.2, 9.3**
    - Use `hypothesis` to generate out-of-bounds `length`, `temperature`, and `tempo` values; assert API returns HTTP 422

  - [ ]* 9.7 Write property test for unauthenticated requests rejected
    - **Property 14: Unauthenticated Requests Rejected**
    - **Validates: Requirements 13.1, 13.2**
    - Use `hypothesis` to generate arbitrary endpoint paths and missing/malformed tokens; assert API returns HTTP 401

  - [ ]* 9.8 Write property test for rate limit enforcement
    - **Property 15: Rate Limit Enforcement**
    - **Validates: Requirements 13.4, 13.5**
    - Use `hypothesis` to simulate >10 POST `/generate` requests within 60 seconds for a single user; assert all requests beyond the tenth return HTTP 429

  - [ ]* 9.9 Write integration tests for FastAPI endpoints
    - Use `httpx.AsyncClient` with `TestClient` against a mock `SupabaseClient`
    - Test full generation flow: valid request → MIDI bytes → signed URL → `TrackRecord` response
    - Test track listing, retrieval, and deletion
    - Test 503 when model not loaded, 502 on storage failure, 401 on missing token
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 10.1, 10.2, 10.3, 13.1, 13.2_

- [ ] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each pipeline stage
- Property tests use `hypothesis` and validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The model must be loaded at API startup; the `/health` endpoint can serve as a readiness probe
