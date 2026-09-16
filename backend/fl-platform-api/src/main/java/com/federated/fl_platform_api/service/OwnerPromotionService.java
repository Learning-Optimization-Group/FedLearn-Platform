package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.NotificationDto;
import com.federated.fl_platform_api.dto.OwnerRequestDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.AccessRequestStatus;
import com.federated.fl_platform_api.model.OwnerPromotionRequest;
import com.federated.fl_platform_api.model.PlatformRole;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.OwnerPromotionRequestRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Owner-promotion workflow: a {@link PlatformRole#USER} requests to become a
 * {@link PlatformRole#PROJECT_OWNER}; a platform admin approves (flips the
 * platform role) or denies. Mirrors {@link AccessRequestService} so the two
 * approval flows behave identically. One row per user (re-request updates it).
 */
@Service
public class OwnerPromotionService {

    @Autowired private OwnerPromotionRequestRepository requestRepository;
    @Autowired private UserRepository userRepository;
    @Autowired private AuthorizationService authz;
    @Autowired private NotificationService notifications;

    @Transactional
    public OwnerRequestDto submit(String message) {
        User caller = authz.currentUser();

        if (caller.getPlatformRole() == PlatformRole.PROJECT_OWNER
                || caller.getPlatformRole() == PlatformRole.PLATFORM_ADMIN) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "You can already create projects");
        }

        // Upsert: a re-request after a DENY reuses the same row (UNIQUE(user_id)).
        OwnerPromotionRequest req = requestRepository.findByUserId(caller.getId())
            .orElseGet(() -> new OwnerPromotionRequest(caller, message));
        if (req.getStatus() == AccessRequestStatus.PENDING && req.getId() != null) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "You already have a pending owner request");
        }
        req.setMessage(message);
        req.setStatus(AccessRequestStatus.PENDING);
        req.setRequestedAt(Instant.now());
        req.setDecidedAt(null);
        req.setDecidedBy(null);
        OwnerPromotionRequest saved = requestRepository.save(req);

        NotificationDto n = new NotificationDto();
        n.setType(NotificationDto.Type.OWNER_PROMOTION_REQUESTED);
        n.setActorId(caller.getId());
        n.setActorUsername(caller.getUsername());
        n.setSubjectId(caller.getId());
        n.setSubjectUsername(caller.getUsername());
        notifications.notifyPlatformAdmins(n);

        return toDto(saved);
    }

    /** The caller's own request, if any (drives the "Request owner access" UI state). */
    public Optional<OwnerRequestDto> getMine() {
        User caller = authz.currentUser();
        return requestRepository.findByUserId(caller.getId()).map(this::toDto);
    }

    /** Admin queue. {@code filter} null returns all. */
    public List<OwnerRequestDto> listForAdmin(AccessRequestStatus filter) {
        authz.requirePlatformAdmin();
        List<OwnerPromotionRequest> rows = (filter != null)
            ? requestRepository.findByStatus(filter)
            : requestRepository.findAll();
        return rows.stream().map(this::toDto).collect(Collectors.toList());
    }

    @Transactional
    public OwnerRequestDto decide(Long requestId, AccessRequestStatus decision) {
        authz.requirePlatformAdmin();
        if (decision != AccessRequestStatus.APPROVED && decision != AccessRequestStatus.DENIED) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                "decision must be APPROVED or DENIED");
        }
        OwnerPromotionRequest req = requestRepository.findById(requestId)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("OwnerPromotionRequest", requestId));
        if (req.getStatus() != AccessRequestStatus.PENDING) {
            throw new ResponseStatusException(HttpStatus.CONFLICT,
                "Request has already been decided");
        }

        User actor = authz.currentUser();
        req.setStatus(decision);
        req.setDecidedAt(Instant.now());
        req.setDecidedBy(actor);

        if (decision == AccessRequestStatus.APPROVED) {
            User target = req.getUser();
            // Don't demote an admin who somehow had a pending request.
            if (target.getPlatformRole() == PlatformRole.USER) {
                target.setPlatformRole(PlatformRole.PROJECT_OWNER);
                userRepository.save(target);
            }
        }
        OwnerPromotionRequest saved = requestRepository.save(req);

        NotificationDto n = new NotificationDto();
        n.setType(NotificationDto.Type.OWNER_PROMOTION_DECIDED);
        n.setActorId(actor.getId());
        n.setActorUsername(actor.getUsername());
        n.setSubjectId(req.getUser().getId());
        n.setSubjectUsername(req.getUser().getUsername());
        n.setDecision(decision.name());
        notifications.notifyUser(req.getUser().getId(), n);

        return toDto(saved);
    }

    private OwnerRequestDto toDto(OwnerPromotionRequest r) {
        OwnerRequestDto d = new OwnerRequestDto();
        d.setId(r.getId());
        d.setUserId(r.getUser().getId());
        d.setUsername(r.getUser().getUsername());
        d.setEmail(r.getUser().getEmail());
        d.setStatus(r.getStatus().name());
        d.setMessage(r.getMessage());
        d.setRequestedAt(r.getRequestedAt());
        d.setDecidedAt(r.getDecidedAt());
        d.setDecidedByUsername(r.getDecidedBy() != null ? r.getDecidedBy().getUsername() : null);
        return d;
    }
}
