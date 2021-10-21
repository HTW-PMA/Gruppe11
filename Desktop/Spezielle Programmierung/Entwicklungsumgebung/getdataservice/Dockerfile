FROM python:alpine3.14
LABEL Zouia Alkurdi <aklkurdiz@htw-berlin.de>

# Install need packages
RUN apk update && \
    apk upgrade && \
    apk add \
        curl \
        jq \
        bash \
    && rm -rf /var/cache/apk/*

# Create folder for data
RUN mkdir app

# Create non-root user
RUN addgroup -S htw-group && \
    adduser -S htw-user -G htw-group --uid 1001

# Change the user
# USER 1001

# Copy data to image
COPY --chown=htw-user:htw-group dax-aktien.txt /app
COPY --chown=htw-user:htw-group getData.sh /app

# Make scripts runable
RUN chmod 755 /app/getData.sh

CMD ./app/getData.sh
# CMD tail -f /dev/null