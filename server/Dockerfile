FROM maven:3.9-eclipse-temurin-17 AS build

WORKDIR /app

COPY pom.xml .
RUN mvn dependency:go-offline -B

COPY src ./src

RUN mvn clean package -DskipTests -B

FROM eclipse-temurin:17-jre AS runtime

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system spring && adduser --system --ingroup spring spring

COPY --from=build /app/target/*.jar app.jar

RUN chown -R spring:spring /app

USER spring

EXPOSE 8080

ENV JAVA_OPTS="-Xmx512m -Xms256m"

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
