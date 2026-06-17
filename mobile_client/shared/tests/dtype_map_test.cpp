// dtype_map_test.cpp — DtypeMap whitelist + hard-error contract (15-LLD §13 task 5, E7).
// Torch-free: validates fedlearn::DType (no torch ScalarType).

#include "fedlearn/DtypeMap.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

TEST(DtypeMap, RoundTripsEverySafeDtype) {
  const std::vector<std::string> safe = {
      "float16", "float32", "float64", "int8",  "int16",
      "int32",   "int64",   "uint8",   "bool",  "bfloat16",
  };
  for (const auto& name : safe) {
    EXPECT_TRUE(fedlearn::isSafeDtype(name)) << name;
    const fedlearn::DType d = fedlearn::dtypeFromString(name);
    EXPECT_EQ(fedlearn::stringFromDtype(d), name) << name;
  }
}

TEST(DtypeMap, UnknownDtypeIsHardError) {
  EXPECT_FALSE(fedlearn::isSafeDtype("object"));
  EXPECT_THROW(fedlearn::dtypeFromString("object"), std::invalid_argument);
  EXPECT_THROW(fedlearn::dtypeFromString("complex64"), std::invalid_argument);
  EXPECT_THROW(fedlearn::dtypeFromString(""), std::invalid_argument);
}
