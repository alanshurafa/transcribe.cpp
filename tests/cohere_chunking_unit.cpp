// Unit test for the cohere long-form split and join (arch/cohere/chunking.h).
// Expected cut points for the synthetic signal come from a transcription of
// transformers CohereAsrFeatureExtractor._split_audio_chunks_energy run on
// the same samples; the port must match exactly. A fake decoder then checks
// that split + join loses and repeats nothing.

#include "arch/cohere/chunking.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ch = transcribe::cohere;

namespace {

constexpr int64_t kSr = 16000;

int g_failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// 32-bit LCG noise with a per-1600-sample amplitude level of 1..64 (64ths).
// Every sample is exactly representable in float, so the reference script
// sees the same values.
std::vector<float> synth(int64_t n) {
    std::vector<float> w(static_cast<size_t>(n));
    uint32_t           state = 12345;
    double             amp   = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (i % 1600 == 0) {
            state = 1664525u * state + 1013904223u;
            amp   = static_cast<double>((state >> 16) % 64 + 1) / 64.0;
        }
        state        = 1664525u * state + 1013904223u;
        const int v  = static_cast<int>((state >> 16) & 0xFFFF) - 32768;
        w[size_t(i)] = static_cast<float>(amp * v / 32768.0);
    }
    return w;
}

bool contiguous(const std::vector<ch::Chunk> & c, int64_t n, int64_t max_len) {
    int64_t pos = 0;
    for (const auto & k : c) {
        if (k.start != pos || k.n_samples < 1 || k.n_samples > max_len) {
            return false;
        }
        pos += k.n_samples;
    }
    return pos == n;
}

// Fake decoder: one word per second of audio, owned by the chunk holding
// that second's first sample, with the leading space a sentencepiece decode
// leaves and a trailing space for good measure.
std::string fake_decode(const ch::Chunk & c) {
    std::string out;
    for (int64_t s = (c.start + kSr - 1) / kSr; s * kSr < c.start + c.n_samples; ++s) {
        out += " w" + std::to_string(s);
    }
    return out.empty() ? out : out + " ";
}

}  // namespace

int main() {
    const ch::ChunkParams p;  // reference defaults: 35 s chunks, 5 s search, 1600-sample windows

    // Up to 35 s is one chunk; one sample more splits.
    {
        std::vector<float> w(size_t(35 * kSr), 0.1f);
        auto               c = ch::split_long_audio(w.data(), int64_t(w.size()), p);
        check(c.size() == 1 && c[0].start == 0 && c[0].n_samples == 35 * kSr, "35 s is one chunk");
        w.push_back(0.1f);
        c = ch::split_long_audio(w.data(), int64_t(w.size()), p);
        check(c.size() == 2 && contiguous(c, int64_t(w.size()), p.chunk_samples), "35 s + 1 sample splits");
        check(ch::split_long_audio(nullptr, 10, p).empty() && ch::split_long_audio(w.data(), 0, p).empty(),
              "empty input gives no chunks");
    }

    // Constant signal: every window ties, the first (search start) wins, so
    // cuts land every 30 s.
    {
        std::vector<float> w(size_t(100 * kSr), 0.25f);
        const auto         c = ch::split_long_audio(w.data(), int64_t(w.size()), p);
        check(c.size() == 4 && c[1].start == 30 * kSr && c[2].start == 60 * kSr && c[3].start == 90 * kSr,
              "flat signal cuts at each search start");
    }

    // Silence inside the search window is where the cut lands.
    {
        std::vector<float> w(size_t(100 * kSr), 0.25f);
        for (int64_t i = 512000; i < 512000 + 3200; ++i) {
            w[size_t(i)] = 0.0f;  // 32.0-32.2 s
        }
        for (int64_t i = 1056000; i < 1056000 + 1600; ++i) {
            w[size_t(i)] = 0.0f;  // 66.0-66.1 s, inside the second search window
        }
        const auto c = ch::split_long_audio(w.data(), int64_t(w.size()), p);
        check(c.size() == 3 && c[1].start == 512000 && c[2].start == 1056000, "cuts land in the silences");
    }

    // Synthetic noise against the reference implementation's cut points.
    {
        const int64_t              n    = 600 * kSr + 12345;
        const std::vector<float>   w    = synth(n);
        const std::vector<int64_t> want = { 483200,  1036800, 1529600, 2057600, 2606400, 3144000,
                                            3660800, 4196800, 4702400, 5187200, 5726400, 6222400,
                                            6740800, 7228800, 7771200, 8297600, 8851200, 9364800 };
        const auto                 c    = ch::split_long_audio(w.data(), n, p);
        bool                       same = c.size() == want.size() + 1;
        for (size_t i = 0; same && i < want.size(); ++i) {
            same = c[i + 1].start == want[i];
        }
        check(same, "cut points match the reference on synthetic noise");
        check(contiguous(c, n, p.chunk_samples), "chunks are contiguous and at most 35 s");
        for (size_t i = 0; i + 1 < c.size(); ++i) {
            check(c[i].n_samples >= p.chunk_samples - p.search_samples, "non-final chunks are at least 30 s");
        }

        // Fake decode each chunk and join: every word once, in order.
        std::vector<std::string> texts;
        for (const auto & k : c) {
            texts.push_back(fake_decode(k));
        }
        texts.insert(texts.begin() + 2, "   ");  // an empty chunk decode is dropped
        std::string want_text;
        for (int64_t s = 0; s * kSr < n; ++s) {
            want_text += (s ? " w" : "w") + std::to_string(s);
        }
        const std::string got = ch::join_chunk_texts(texts, "en");
        check(got.substr(1) == want_text && got[0] == ' ', "fake decode joins with no lost or repeated words");
    }

    // Join rules (CohereAsrProcessor._reassemble_chunk_texts).
    check(ch::join_chunk_texts({ "a ", "", " b", " \n", "c" }, "en") == "a b c", "space join drops empty texts");
    check(ch::join_chunk_texts({ "  a ", " b " }, "fr") == "  a b", "first text keeps its left side");
    // "nihon" + "go" and "zhong" + "wen" as UTF-8 bytes (ASCII-only source).
    check(ch::join_chunk_texts({ "\xe6\x97\xa5\xe6\x9c\xac ", " \xe8\xaa\x9e" }, "ja") ==
              "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e",
          "ja joins without a space");
    check(ch::join_chunk_texts({ "\xe4\xb8\xad ", " \xe6\x96\x87" }, "zh") == "\xe4\xb8\xad\xe6\x96\x87",
          "zh joins without a space");
    check(ch::join_chunk_texts({}, "en").empty() && ch::join_chunk_texts({ " " }, nullptr).empty(), "nothing to join");

    if (g_failures == 0) {
        std::printf("cohere chunking unit: OK\n");
    }
    return g_failures == 0 ? 0 : 1;
}
