package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.NotificationDto;
import com.federated.fl_platform_api.model.ProjectMembership;
import com.federated.fl_platform_api.model.MembershipRole;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class NotificationService {

    @Autowired private WebSocketService webSocketService;
    @Autowired private ProjectMembershipRepository membershipRepository;

    public void notifyUser(Long userId, NotificationDto payload) {
        webSocketService.sendUserNotification(userId, payload);
    }

    /**
     * Notify the project's owner + all MEMBER role rows. Used when a new
     * access request is created — the spec routes such events to anyone who
     * can act on them (spec §6.3).
     */
    public void notifyOwnerAndMembers(java.util.UUID projectId, Long ownerId, NotificationDto payload) {
        notifyUser(ownerId, payload);
        List<ProjectMembership> moderators =
            membershipRepository.findByIdProjectIdAndRole(projectId, MembershipRole.MEMBER);
        for (ProjectMembership m : moderators) {
            notifyUser(m.getId().getUserId(), payload);
        }
    }
}
