#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include "fedlearn/Sha256.h"
#include "fedlearn/TrainingStateError.h"

namespace fedlearn {

// The transport owns data until append returns; assembled bytes are owned here.
struct ModelStreamPart {
  int index;
  int totalChunks;
  bool isFinal;
  int currentRound;
  std::string codec;
  bool compressed;
  int64_t totalBytes;
  std::string sha256;
  std::string_view data;
};

class ModelStream {
 public:
  explicit ModelStream(int64_t maxBytes) : maxBytes_(maxBytes) {}

  void append(const ModelStreamPart& part) {
    if (complete_ || part.index != next_ || part.totalChunks <= 0 ||
        part.index >= part.totalChunks || part.isFinal != (part.index == part.totalChunks - 1)) {
      throw TrainingStateError("model download has invalid chunk order or final marker");
    }
    if (part.codec != "safetensors" || part.compressed || part.currentRound < 1 ||
        part.totalBytes <= 0 || part.totalBytes > maxBytes_ || part.totalChunks > part.totalBytes ||
        part.sha256.size() != 64 ||
        !std::all_of(part.sha256.begin(), part.sha256.end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        })) {
      throw TrainingStateError("model download has invalid codec, size, round or checksum metadata");
    }
    if (next_ == 0) {
      round_ = part.currentRound;
      count_ = part.totalChunks;
      bytes_ = part.totalBytes;
      sha_ = part.sha256;
    } else if (part.currentRound != round_ || part.totalChunks != count_ ||
               part.totalBytes != bytes_ || part.sha256 != sha_) {
      throw TrainingStateError("model download metadata changed between chunks");
    }
    if (part.data.empty() || part.data.size() > static_cast<uint64_t>(bytes_) - blob_.size()) {
      throw TrainingStateError("model download exceeds its declared size or contains an empty chunk");
    }
    blob_.append(part.data);
    ++next_;
    complete_ = part.isFinal;
    if (complete_ && blob_.size() != static_cast<uint64_t>(bytes_)) {
      throw TrainingStateError("model download length differs from its declared size");
    }
  }

  std::string finish() {
    if (!complete_ || taken_ || Sha256::hexDigest(blob_) != sha_) {
      throw TrainingStateError("model download is incomplete or its checksum does not match");
    }
    taken_ = true;
    return std::move(blob_);
  }

  int currentRound() const { return round_; }

 private:
  int64_t maxBytes_;
  int next_ = 0;
  int count_ = 0;
  int round_ = 0;
  int64_t bytes_ = 0;
  std::string sha_;
  bool complete_ = false;
  bool taken_ = false;
  std::string blob_;
};

}  // namespace fedlearn
