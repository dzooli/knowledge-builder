FROM ghcr.io/paperless-ngx/paperless-ngx:latest

# Ensure we run from the Django project root
WORKDIR /usr/src/paperless/src

# Faster, cleaner Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy entrypoint script (LF normalized)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run the token creation entrypoint
ENTRYPOINT ["/bin/sh", "-lc", "/entrypoint.sh"]

