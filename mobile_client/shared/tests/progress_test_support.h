#pragma once

#include <cstdint>
#include <string>
#include <tuple>

namespace fedtest {
using Progress = std::tuple<std::string, int, int64_t, int64_t>;

template <typename Client>
Progress progressOf(const Client& client) {
  const auto p = client.progressSnapshot();
  return {p.status, p.round, p.currentStep, p.totalSteps};
}

template <typename Client>
void publishTo(Client& client, const Progress& value) {
  client.updateProgress(std::get<0>(value), std::get<1>(value), std::get<2>(value), std::get<3>(value));
}
}  // namespace fedtest
