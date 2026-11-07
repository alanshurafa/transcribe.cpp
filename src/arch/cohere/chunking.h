// arch/cohere/chunking.h - long-form input splitting for Cohere ASR.
//
// The decoder is trained on clips of at most max_audio_clip_s (35 s). Fed a
// longer clip in one pass it drops or loops content without hitting EOS
// late, so nothing reports truncation (issue #128). The reference processor
// (transformers CohereAsrFeatureExtractor._split_audio_chunks_energy) splits
// longer audio before the encoder and joins the per-chunk texts; this is a
// port of that split and of CohereAsrProcessor._reassemble_chunk_texts.
//
// Split: walk the audio in steps of at most chunk_samples. Each cut lands at
// the start of the quietest (lowest RMS) non-overlapping window of
// window_samples inside the last search_samples before the step limit.
// Chunks are contiguous and never overlap. Host-only; no ggml.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace transcribe::cohere {

struct ChunkParams {
    // Reference feature-extractor defaults (max_audio_clip_s = 35,
    // overlap_chunk_second = 5, min_energy_window_samples = 1600) at 16 kHz.
    int64_t chunk_samples  = 35 * 16000;
    int64_t search_samples = 5 * 16000;
    int64_t window_samples = 1600;
};

struct Chunk {
    int64_t start     = 0;  // first sample
    int64_t n_samples = 0;
};

// One chunk covering everything when n_samples <= chunk_samples; otherwise
// contiguous chunks of at most chunk_samples each.
std::vector<Chunk> split_long_audio(const float * pcm, int64_t n_samples, const ChunkParams & p);

// Join per-chunk texts in order: drop empty or all-whitespace texts, trim
// the right of the first and both sides of the rest, and join with a space,
// or with nothing for languages written without spaces (ja, zh).
std::string join_chunk_texts(const std::vector<std::string> & texts, const char * language);

}  // namespace transcribe::cohere
