package com.federated.fl_platform_api.repository;

import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.OwnerPromotionRequest;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface OwnerPromotionRequestRepository extends JpaRepository<OwnerPromotionRequest, Long> {

    Optional<OwnerPromotionRequest> findByUserId(Long userId);

    List<OwnerPromotionRequest> findByStatus(AccessRequestStatus status);

    long countByStatus(AccessRequestStatus status);
}
