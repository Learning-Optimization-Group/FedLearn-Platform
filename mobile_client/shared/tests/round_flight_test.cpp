#include "fedlearn/RoundFlight.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(RoundFlight, RejectsOverlapAndReleasesOnScopeExit) {
  fedlearn::RoundFlight flight;
  {
    auto active = flight.acquire();
    EXPECT_THROW(flight.acquire(), std::runtime_error);
  }
  EXPECT_NO_THROW(flight.acquire());
}

TEST(RoundFlight, ReleasesWhenTrainingThrows) {
  fedlearn::RoundFlight flight;
  EXPECT_THROW(([&]() {
    auto active = flight.acquire();
    throw std::runtime_error("round failed");
  })(), std::runtime_error);
  EXPECT_NO_THROW(flight.acquire());
}
