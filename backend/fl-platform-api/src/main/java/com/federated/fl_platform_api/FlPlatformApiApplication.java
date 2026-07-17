package com.federated.fl_platform_api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@ConfigurationPropertiesScan
@EnableScheduling   // finding 4: drives StartupReconciler's periodic stuck-run sweep
public class FlPlatformApiApplication {

	public static void main(String[] args) {
		SpringApplication.run(FlPlatformApiApplication.class, args);
	}

}
