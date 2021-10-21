FROM ubuntu:21.04
LABEL Zouia Alkurdi <aklkurdiz@htw-berlin.de>

# Update system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y

# Install time package
RUN apt-get install -y \
        tzdata \
        iputils-ping

# Install python and python components
RUN apt-get install -y \ 
        python3-pip \
        python3-numpy \
        python3-pandas

# Cleanup
RUN apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/cache/apk/*

# Install scikit-learn
RUN pip3 --no-cache-dir install \
    scikit-learn \
    minio

# Create app dir
RUN mkdir app

# Create non-root user
RUN addgroup htw-group && \
    adduser htw-user --disabled-password --gecos "" --uid 1001 && \
    usermod -aG htw-group htw-user

# Change the user
# USER 1001

# Copy data to image
COPY --chown=htw-user:htw-group forecast.py /app

# Start forecast
CMD python3 /app/forecast.py
#CMD tail -f /dev/null