# =============================================================================
# FedLearn Platform API — Docker image
# =============================================================================
# Stage 1: build the Spring Boot jar with the project's Gradle wrapper.
# Stage 2: assemble a slim JRE runtime that runs as a non-root user.
# =============================================================================

FROM eclipse-temurin:21-jdk AS build
WORKDIR /app

# Copy Gradle wrapper and config first so dependency resolution can be cached
# independently of source changes.
COPY gradlew .
COPY gradle gradle
COPY settings.gradle build.gradle ./
RUN chmod +x gradlew && ./gradlew --no-daemon dependencies || true

COPY src src
RUN ./gradlew --no-daemon bootJar -x test

FROM eclipse-temurin:21-jre
WORKDIR /app

# Unprivileged runtime user.
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --system --gid ${APP_GID} fedlearn \
    && useradd --system --uid ${APP_UID} --gid ${APP_GID} --no-create-home --shell /usr/sbin/nologin fedlearn

COPY --from=build /app/build/libs/*.jar /app/app.jar

EXPOSE 8081

# Entrypoint:
#   - Rewrites Heroku/RDS-style DATABASE_URL (postgres://) to JDBC form.
#   - Fails fast if required secrets are missing from the orchestrator env.
COPY <<'EOF' /app/entrypoint.sh
#!/bin/sh
set -eu

if [ -n "${DATABASE_URL:-}" ]; then
  export SPRING_DATASOURCE_URL=$(echo "$DATABASE_URL" | sed 's|^postgres://|jdbc:postgresql://|')
fi

required_vars="APP_JWT_SECRET APP_INTERNAL_API_KEY SPRING_DATASOURCE_URL SPRING_DATASOURCE_USERNAME SPRING_DATASOURCE_PASSWORD"
missing=""
for v in $required_vars; do
  eval "val=\${$v:-}"
  if [ -z "$val" ]; then
    missing="$missing $v"
  fi
done
if [ -n "$missing" ]; then
  echo "ERROR: Required environment variables are missing:$missing" >&2
  exit 1
fi

exec java -jar /app/app.jar "$@"
EOF
RUN chmod +x /app/entrypoint.sh \
    && chown -R fedlearn:fedlearn /app

USER fedlearn
ENTRYPOINT ["/app/entrypoint.sh"]
