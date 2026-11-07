// arch/cohere/chunking.cpp - see chunking.h.

#include "chunking.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace transcribe::cohere {

namespace {

// _find_split_point_energy: start of the quietest window_samples window in
// [start, end), windows stepped by their own length; the midpoint when the
// range is no longer than one window.
int64_t find_split_point_energy(const float * pcm, int64_t start, int64_t end, int64_t window) {
    const int64_t len = end - start;
    if (len <= window) {
        return (start + end) / 2;
    }
    double  min_energy = std::numeric_limits<double>::infinity();
    int64_t quietest   = start;
    for (int64_t i = 0; i < len - window; i += window) {
        double sum = 0.0;
        for (int64_t k = 0; k < window; ++k) {
            const double x = pcm[start + i + k];
            sum += x * x;
        }
        const double energy = std::sqrt(sum / static_cast<double>(window));
        if (energy < min_energy) {
            min_energy = energy;
            quietest   = start + i;
        }
    }
    return quietest;
}

bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string trim(const std::string & s, bool left) {
    size_t b = 0;
    size_t e = s.size();
    if (left) {
        while (b < e && is_space(s[b])) {
            ++b;
        }
    }
    while (e > b && is_space(s[e - 1])) {
        --e;
    }
    return s.substr(b, e - b);
}

}  // namespace

std::vector<Chunk> split_long_audio(const float * pcm, int64_t n_samples, const ChunkParams & p) {
    std::vector<Chunk> chunks;
    if (pcm == nullptr || n_samples <= 0) {
        return chunks;
    }
    const int64_t chunk  = std::max<int64_t>(1, p.chunk_samples);
    const int64_t search = std::max<int64_t>(1, p.search_samples);
    const int64_t window = std::max<int64_t>(1, p.window_samples);

    int64_t idx = 0;
    while (idx < n_samples) {
        if (idx + chunk >= n_samples) {
            chunks.push_back({ idx, n_samples - idx });
            break;
        }
        const int64_t search_start = std::max(idx, idx + chunk - search);
        const int64_t search_end   = std::min(idx + chunk, n_samples);
        int64_t       split        = idx + chunk;
        if (search_end > search_start) {
            split = find_split_point_energy(pcm, search_start, search_end, window);
        }
        split = std::max(idx + 1, std::min(split, n_samples));
        chunks.push_back({ idx, split - idx });
        idx = split;
    }
    return chunks;
}

std::string join_chunk_texts(const std::vector<std::string> & texts, const char * language) {
    const bool no_space = language != nullptr && (std::strcmp(language, "ja") == 0 || std::strcmp(language, "zh") == 0);
    const char * separator = no_space ? "" : " ";
    std::string  out;
    bool         first = true;
    for (const std::string & t : texts) {
        if (trim(t, /*left=*/true).empty()) {
            continue;
        }
        if (first) {
            out   = trim(t, /*left=*/false);
            first = false;
        } else {
            out += separator;
            out += trim(t, /*left=*/true);
        }
    }
    return out;
}

}  // namespace transcribe::cohere
