package com.federated.fl_platform_api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class FlPlatformApiApplication {

	public static void main(String[] args) {
		SpringApplication.run(FlPlatformApiApplication.class, args);
	}

}
