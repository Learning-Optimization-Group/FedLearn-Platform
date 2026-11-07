package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.RoundResultDto;
import com.federated.fl_platform_api.model.RoundResult;
import com.federated.fl_platform_api.repository.RoundResultRepository;
import com.federated.fl_platform_api.security.InternalCallbackAuthorization;
import com.federated.fl_platform_api.security.InternalCallbackContext;
import java.util.Objects;
import java.util.UUID;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class RoundResultIngestService {
    public record Recorded(RoundResult result, boolean created) {}

    private final InternalCallbackAuthorization authorization;
    private final RoundResultRepository results;

    public RoundResultIngestService(InternalCallbackAuthorization authorization, RoundResultRepository results) {
        this.authorization = authorization;
        this.results = results;
    }

    @Transactional
    public Recorded record(UUID projectId, InternalCallbackContext context, RoundResultDto dto) {
        var authorized = authorization.requireActiveRun(projectId, context);
        var project = authorized.project();
        var run = authorized.run();
        if (dto.getServerRound() == null || dto.getServerRound() < 1 || dto.getServerRound() > run.getNumRounds()) {
            throw new IllegalArgumentException("Round must be within the run's round budget");
        }
        requireFinite(dto.getLoss());
        requireFinite(dto.getAccuracy());
        requireFinite(dto.getGpuUtilization());
        requireFinite(dto.getTimeDataLoadingS());
        requireFinite(dto.getTimeForwardBackwardS());
        requireFinite(dto.getTimeOptimizerStepS());
        requireFinite(dto.getTimeWireEncodeS());
        requireFinite(dto.getTimeAggregationS());
        requireFinite(dto.getTimeWireDecodeS());
        requireFinite(dto.getTimeEvalS());
        var existing = results.findByRunIdAndServerRound(run.getId(), dto.getServerRound());
        if (existing.isPresent()) {
            var previous = existing.get();
            if (!Objects.equals(previous.getLoss(), dto.getLoss())
                    || !Objects.equals(previous.getAccuracy(), dto.getAccuracy())
                    || !Objects.equals(previous.getGpuUtilization(), dto.getGpuUtilization())
                    || !Objects.equals(previous.getTimeDataLoadingS(), dto.getTimeDataLoadingS())
                    || !Objects.equals(previous.getTimeForwardBackwardS(), dto.getTimeForwardBackwardS())
                    || !Objects.equals(previous.getTimeOptimizerStepS(), dto.getTimeOptimizerStepS())
                    || !Objects.equals(previous.getTimeWireEncodeS(), dto.getTimeWireEncodeS())
                    || !Objects.equals(previous.getTimeAggregationS(), dto.getTimeAggregationS())
                    || !Objects.equals(previous.getTimeWireDecodeS(), dto.getTimeWireDecodeS())
                    || !Objects.equals(previous.getTimeEvalS(), dto.getTimeEvalS())) {
                throw new IllegalStateException("Round result conflicts with the recorded metrics");
            }
            return new Recorded(previous, false);
        }
        var result = new RoundResult();
        result.setProject(project);
        result.setRunId(run.getId());
        result.setServerRound(dto.getServerRound());
        result.setLoss(dto.getLoss());
        result.setAccuracy(dto.getAccuracy());
        result.setGpuUtilization(dto.getGpuUtilization());
        result.setTimeDataLoadingS(dto.getTimeDataLoadingS());
        result.setTimeForwardBackwardS(dto.getTimeForwardBackwardS());
        result.setTimeOptimizerStepS(dto.getTimeOptimizerStepS());
        result.setTimeWireEncodeS(dto.getTimeWireEncodeS());
        result.setTimeAggregationS(dto.getTimeAggregationS());
        result.setTimeWireDecodeS(dto.getTimeWireDecodeS());
        result.setTimeEvalS(dto.getTimeEvalS());
        return new Recorded(results.saveAndFlush(result), true);
    }

    private static void requireFinite(Double value) {
        if (value != null && !Double.isFinite(value)) {
            throw new IllegalArgumentException("Round metrics must be finite or null");
        }
    }
}
