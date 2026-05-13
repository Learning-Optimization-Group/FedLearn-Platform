package com.federated.fl_platform_api.client;

import com.federated.fl_platform_api.model.*;
import com.federated.fl_platform_api.repository.*;
import com.federated.fl_platform_api.service.ClientApiService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.*;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
// BEFORE_CLASS forces a fresh context (and H2 schema recreation) before this class runs.
// Without it, a prior AFTER_EACH context teardown from another test class can drop the
// shared H2 in-memory schema, leaving this context's queries with no tables.
@DirtiesContext(classMode = DirtiesContext.ClassMode.BEFORE_CLASS)
class PartitionAssignmentConcurrencyTest {

    @Autowired ClientApiService clientApiService;
    @Autowired ProjectRepository projectRepository;
    @Autowired UserRepository userRepository;
    @Autowired ProjectMembershipRepository membershipRepository;
    @Autowired PasswordEncoder passwordEncoder;

    @Test
    void concurrentConnections_yieldDistinctPartitionIds() throws Exception {
        User owner = userRepository.save(new User("owner_p", "owner_p@example.com",
            passwordEncoder.encode("Password1!")));
        Project p = new Project();
        p.setName("conc-" + System.nanoTime());
        p.setModelType("CNN-CIFAR10");
        p.setModelName("resnet8");
        p.setStatus("RUNNING");
        p.setServerPort(50000);
        p.setUser(owner);
        p.setVisibility(ProjectVisibility.PRIVATE);
        projectRepository.saveAndFlush(p);

        int n = 10;
        List<User> clients = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            User u = userRepository.save(new User("c" + i,
                "c" + i + "@example.com", passwordEncoder.encode("Password1!")));
            membershipRepository.save(new ProjectMembership(
                p, u, MembershipRole.CLIENT, JoinedVia.OWNER_ADD, owner));
            clients.add(u);
        }

        ExecutorService pool = Executors.newFixedThreadPool(n);
        CountDownLatch ready = new CountDownLatch(n);
        CountDownLatch go = new CountDownLatch(1);
        List<Future<Integer>> futures = new ArrayList<>();
        for (User u : clients) {
            futures.add(pool.submit(() -> {
                SecurityContextHolder.getContext().setAuthentication(
                    new UsernamePasswordAuthenticationToken(
                        u.getUsername(), "x",
                        List.of(new SimpleGrantedAuthority("ROLE_USER"))));
                ready.countDown();
                go.await();
                return clientApiService.getConnection(p.getId()).getPartitionId();
            }));
        }
        ready.await(5, TimeUnit.SECONDS);
        go.countDown();
        pool.shutdown();
        pool.awaitTermination(15, TimeUnit.SECONDS);

        Set<Integer> assigned = new HashSet<>();
        for (Future<Integer> f : futures) {
            assigned.add(f.get(5, TimeUnit.SECONDS));
        }
        assertEquals(n, assigned.size(),
            "Each concurrent connection must receive a distinct partition_id");
    }
}
