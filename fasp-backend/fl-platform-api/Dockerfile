
# Stage 1 - Build the Spring Boot application using Maven


FROM maven:3.9-eclipse-temurin-21 AS build
WORKDIR /app

# Copy the Maven project definition first to leverage Docker's layer caching
COPY pom.xml .
COPY .mvn .mvn

# Copy the rest of the source code
COPY src src

# Build the application, skipping tests to speed up the process
RUN mvn package -DskipTests

# Stage 2 - Create the Python virtual environment

FROM python:3.10-slim AS python-env
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Create a virtual environment and install dependencies into it

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3 - Assemble the final image

FROM openjdk:21-jdk-slim
WORKDIR /app

# Copy the python environment
# Copy the virtual environment from the python-env stage
COPY --from=python-env /opt/venv /opt/venv

# --- Copy the Python Scripts ---
# Create a directory for the scripts and copy them in
RUN mkdir -p /app/scripts
COPY src/main/resources/scripts /app/scripts

# --- Copy the Java Application ---
# Copy the built JAR file from the build stage
COPY --from=build /app/target/*.jar app.jar

# --- Configure Environment ---
# Add the venv's bin directory to the system's PATH
# This ensures the 'python' command uses the venv's interpreter
ENV PATH="/opt/venv/bin:$PATH"

# Expose the application port
EXPOSE 8081

# --- Set the entrypoint to run the application ---
# This command will be executed when the container starts
ENTRYPOINT ["java", "-jar", "/app/app.jar"]

