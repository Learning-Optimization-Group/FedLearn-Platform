package com.federated.fl_platform_api.testsupport;

import org.springframework.lang.Nullable;
import org.springframework.test.context.ContextConfigurationAttributes;
import org.springframework.test.context.ContextCustomizer;
import org.springframework.test.context.ContextCustomizerFactory;

import java.util.List;

/**
 * SPI entry point for {@link PerContextDatabaseIsolationCustomizer}. Registered in
 * {@code src/test/resources/META-INF/spring.factories} under
 * {@code org.springframework.test.context.ContextCustomizerFactory}, so the customizer is attached
 * to every test context without editing any test class.
 *
 * <p>Returns a shared singleton customizer so the TestContext cache key is unchanged (all contexts
 * carry an equal customizer): cache-equal test classes keep sharing one context and one isolated
 * database; cache-distinct contexts each get their own.
 */
public class PerContextDatabaseIsolationCustomizerFactory implements ContextCustomizerFactory {

    @Override
    @Nullable
    public ContextCustomizer createContextCustomizer(Class<?> testClass,
                                                     List<ContextConfigurationAttributes> configAttributes) {
        return PerContextDatabaseIsolationCustomizer.INSTANCE;
    }
}
