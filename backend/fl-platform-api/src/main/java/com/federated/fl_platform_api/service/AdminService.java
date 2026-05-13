package com.federated.fl_platform_api.service;

import com.federated.fl_platform_api.dto.AdminUserDto;
import com.federated.fl_platform_api.dto.ProjectResponseDto;
import com.federated.fl_platform_api.exception.ResourceNotFoundException;
import com.federated.fl_platform_api.model.User;
import com.federated.fl_platform_api.repository.ProjectMembershipRepository;
import com.federated.fl_platform_api.repository.ProjectRepository;
import com.federated.fl_platform_api.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class AdminService {

    @Autowired private UserRepository userRepository;
    @Autowired private ProjectRepository projectRepository;
    @Autowired private ProjectMembershipRepository membershipRepository;

    public List<AdminUserDto> listUsers() {
        return userRepository.findAll().stream().map(this::toDto).collect(Collectors.toList());
    }

    public AdminUserDto getUser(Long id) {
        User u = userRepository.findById(id)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", id));
        return toDto(u);
    }

    @Transactional
    public AdminUserDto updateRole(Long id, String newRole) {
        User target = userRepository.findById(id)
            .orElseThrow(() -> ResourceNotFoundException.forEntity("User", id));

        if ("USER".equals(newRole) && "ADMIN".equals(target.getRole())) {
            long adminCount = userRepository.countByRole("ADMIN");
            if (adminCount <= 1) {
                throw new ResponseStatusException(HttpStatus.CONFLICT,
                    "Cannot demote the only remaining admin");
            }
        }
        target.setRole(newRole);
        return toDto(userRepository.save(target));
    }

    public List<ProjectResponseDto> listAllProjects() {
        return projectRepository.findAll().stream().map(p -> {
            ProjectResponseDto d = new ProjectResponseDto();
            d.setId(p.getId());
            d.setName(p.getName());
            d.setModelType(p.getModelType());
            d.setModelName(p.getModelName());
            d.setServerPort(p.getServerPort());
            d.setOptimizer(p.getOptimizer());
            d.setStatus(p.getStatus());
            d.setVisibility(p.getVisibility() != null ? p.getVisibility().name() : null);
            return d;
        }).collect(Collectors.toList());
    }

    private AdminUserDto toDto(User u) {
        AdminUserDto d = new AdminUserDto();
        d.setId(u.getId());
        d.setUsername(u.getUsername());
        d.setEmail(u.getEmail());
        d.setRole(u.getRole());
        d.setProjectsOwned(projectRepository.findByUserId(u.getId()).size());
        d.setMemberships(membershipRepository.findByIdUserId(u.getId()).size());
        d.setCreatedAt(u.getCreatedAt());
        return d;
    }
}
