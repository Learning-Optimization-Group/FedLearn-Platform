#include "fedlearn/Safetensors.h"

#include <cstring>
#include <stdexcept>

namespace fedlearn {

std::string saveSafetensors(const std::vector<NamedTensor>& tensors, const MetadataList& metadata) {
  std::string header = "{";
  std::string data;
  uint64_t off = 0;
  for (size_t t = 0; t < tensors.size(); ++t) {
    const NamedTensor& nt = tensors[t];
    if (t != 0) header += ",";
    header += "\"" + nt.name + "\":{\"dtype\":\"F32\",\"shape\":[";
    if (nt.shape.empty()) {
      // 0-d scalar -> shape [1] to match the Python safetensors codec byte-for-byte (np.ascontiguousarray
      // yields ndim>=1). Emitting [] would change the header JSON, the u64 header-length prefix, and any
      // golden-vector SHA, breaking wire parity.
      header += "1";
    } else {
      for (size_t i = 0; i < nt.shape.size(); ++i) {
        if (i != 0) header += ",";
        header += std::to_string(nt.shape[i]);
      }
    }
    const uint64_t nbytes = static_cast<uint64_t>(nt.data.size()) * sizeof(float);
    header += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nbytes) + "]}";
    data.append(reinterpret_cast<const char*>(nt.data.data()), static_cast<size_t>(nbytes));
    off += nbytes;
  }
  if (!metadata.empty()) {
    if (!tensors.empty()) header += ",";
    header += "\"__metadata__\":{";
    for (size_t i = 0; i < metadata.size(); ++i) {
      if (i != 0) header += ",";
      header += "\"" + metadata[i].first + "\":\"" + metadata[i].second + "\"";
    }
    header += "}";
  }
  header += "}";

  std::string out;
  const uint64_t hlen = header.size();
  char lenbuf[8];
  for (int i = 0; i < 8; ++i) lenbuf[i] = static_cast<char>((hlen >> (8 * i)) & 0xff);  // little-endian
  out.append(lenbuf, 8);
  out += header;
  out += data;
  return out;
}

namespace {

// Minimal recursive-descent parser for the constrained safetensors header JSON (compact, no
// whitespace): objects, "strings", integer arrays. Sufficient for the F32 state-dict format.
struct Parser {
  const std::string& s;
  size_t i = 0;
  explicit Parser(const std::string& str) : s(str) {}

  char peek() const { return i < s.size() ? s[i] : '\0'; }
  void expect(char c) {
    if (peek() != c) throw std::runtime_error(std::string("safetensors: expected '") + c + "' in header");
    ++i;
  }
  std::string parseString() {
    expect('"');
    std::string out;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.size()) {
        ++i;
        const char c = s[i];
        out += (c == 'n') ? '\n' : (c == 't') ? '\t' : c;
      } else {
        out += s[i];
      }
      ++i;
    }
    expect('"');
    return out;
  }
  int64_t parseInt() {
    const size_t start = i;
    if (peek() == '-') ++i;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i == start) throw std::runtime_error("safetensors: expected integer in header");
    return std::stoll(s.substr(start, i - start));
  }
  std::vector<int64_t> parseIntArray() {
    expect('[');
    std::vector<int64_t> out;
    if (peek() == ']') { ++i; return out; }
    while (true) {
      out.push_back(parseInt());
      if (peek() == ',') { ++i; continue; }
      break;
    }
    expect(']');
    return out;
  }
};

}  // namespace

std::vector<NamedTensor> loadSafetensors(const std::string& blob, MetadataList* metadata) {
  if (blob.size() < 8) throw std::runtime_error("safetensors: blob too short");
  uint64_t hlen = 0;
  for (int i = 0; i < 8; ++i) {
    hlen |= static_cast<uint64_t>(static_cast<unsigned char>(blob[static_cast<size_t>(i)])) << (8 * i);
  }
  // Compare WITHOUT computing 8 + hlen: hlen is an attacker-controlled uint64, so `8 + hlen` can wrap
  // around (e.g. hlen = 2^64-1 -> 7) and pass a `> blob.size()` test. blob.size() >= 8 is guaranteed
  // above, so `hlen > blob.size() - 8` is the overflow-safe form.
  if (hlen > blob.size() - 8) {
    throw std::runtime_error("safetensors: header length exceeds blob (corrupt or legacy pickle blob)");
  }
  const std::string header = blob.substr(8, static_cast<size_t>(hlen));
  const char* dataBase = blob.data() + 8 + hlen;
  const uint64_t dataLen = blob.size() - 8 - hlen;

  if (metadata) metadata->clear();
  std::vector<NamedTensor> tensors;

  Parser p(header);
  p.expect('{');
  if (p.peek() == '}') { ++p.i; return tensors; }
  while (true) {
    const std::string key = p.parseString();
    p.expect(':');
    if (key == "__metadata__") {
      p.expect('{');
      if (p.peek() != '}') {
        while (true) {
          const std::string mk = p.parseString();
          p.expect(':');
          const std::string mv = p.parseString();
          if (metadata) metadata->emplace_back(mk, mv);
          if (p.peek() == ',') { ++p.i; continue; }
          break;
        }
      }
      p.expect('}');
    } else {
      p.expect('{');
      std::string dtype;
      std::vector<int64_t> shape;
      std::vector<int64_t> offsets;
      while (true) {
        const std::string fk = p.parseString();
        p.expect(':');
        if (fk == "dtype") dtype = p.parseString();
        else if (fk == "shape") shape = p.parseIntArray();
        else if (fk == "data_offsets") offsets = p.parseIntArray();
        else throw std::runtime_error("safetensors: unexpected tensor field '" + fk + "'");
        if (p.peek() == ',') { ++p.i; continue; }
        break;
      }
      p.expect('}');
      if (dtype != "F32") throw std::runtime_error("safetensors: only F32 supported, got '" + dtype + "'");
      if (offsets.size() != 2) throw std::runtime_error("safetensors: malformed data_offsets");
      const int64_t s0 = offsets[0], e0 = offsets[1];
      if (s0 < 0 || e0 < s0 || static_cast<uint64_t>(e0) > dataLen) {
        throw std::runtime_error("safetensors: data offsets out of range");
      }
      NamedTensor nt;
      nt.name = key;
      nt.shape = shape;
      // The byte span MUST be a whole number of floats: resize allocates floor(nbytes/4) floats but the
      // memcpy below copies the full (e0-s0) bytes, so a non-multiple-of-4 span (e.g. data_offsets [0,6])
      // would write past the vector storage — a heap overflow from a crafted server blob over plaintext
      // gRPC. Reject it (the Python codec rejects the same input).
      if ((e0 - s0) % static_cast<int64_t>(sizeof(float)) != 0) {
        throw std::runtime_error("safetensors: F32 tensor byte length not a multiple of 4");
      }
      nt.data.resize(static_cast<size_t>(e0 - s0) / sizeof(float));
      std::memcpy(nt.data.data(), dataBase + s0, static_cast<size_t>(e0 - s0));
      tensors.push_back(std::move(nt));
    }
    if (p.peek() == ',') { ++p.i; continue; }
    break;
  }
  p.expect('}');
  return tensors;
}

}  // namespace fedlearn
