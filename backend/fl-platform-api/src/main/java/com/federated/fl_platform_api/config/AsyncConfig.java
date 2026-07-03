package com.federated.fl_platform_api.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.task.TaskExecutor;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.ThreadPoolExecutor;

/**
 * BA-1: enables {@code @Async} and provides the bounded executor that runs model initialisation off
 * the request thread ({@link com.federated.fl_platform_api.service.ModelInitializationWorker}).
 *
 * <p>The pool is size-bounded so a burst of project creations cannot spawn unbounded concurrent
 * Python processes. If the queue saturates, {@link ThreadPoolExecutor.CallerRunsPolicy} runs the init
 * on the submitting thread rather than dropping it — graceful degradation to the (still
 * subprocess-timeout-bounded) synchronous path, and never a project stuck in {@code INITIALIZING}
 * because its task was silently rejected.</p>
 */
@Configuration
@EnableAsync
public class AsyncConfig {

    @Bean("modelInitExecutor")
    public TaskExecutor modelInitExecutor(
            @Value("${app.model-init.pool-size:2}") int poolSize,
            @Value("${app.model-init.queue-capacity:100}") int queueCapacity) {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(poolSize);
        executor.setMaxPoolSize(poolSize);
        executor.setQueueCapacity(queueCapacity);
        executor.setThreadNamePrefix("model-init-");
        executor.setRejectedExecutionHandler(new ThreadPoolExecutor.CallerRunsPolicy());
        executor.initialize();
        return executor;
    }
}
