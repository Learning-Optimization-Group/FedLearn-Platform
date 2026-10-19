// model_execution_error_test.cpp — how a native failure crosses the JS bridge. A model-execution error (ExecuTorch
// refused to load or run a program on this device's data) is deterministic, so its rejection carries a prefix the
// training loop reads as "do not retry"; every other failure keeps its message as it was.
#include "fedlearn/ModelExecutionError.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

TEST(ModelExecutionError, IsARuntimeErrorSoExistingHandlersStillCatchIt) {
  EXPECT_THROW(throw fedlearn::ModelExecutionError("x"), std::runtime_error);
}

TEST(ModelExecutionError, ItsRejectionIsPrefixedSoTheLoopDoesNotRetryIt) {
  const fedlearn::ModelExecutionError e("TrainableExecutorchModel: execute_forward_backward failed (error 16)");
  EXPECT_EQ(fedlearn::rejectionMessage(e),
            "MODEL_EXECUTION: TrainableExecutorchModel: execute_forward_backward failed (error 16)");
}

TEST(ModelExecutionError, OtherFailuresKeepTheirMessage) {
  EXPECT_EQ(fedlearn::rejectionMessage(std::runtime_error("UNAVAILABLE: Socket closed")),
            "UNAVAILABLE: Socket closed");
  EXPECT_EQ(fedlearn::rejectionMessage(std::runtime_error("STOP: server finished")), "STOP: server finished");
}
