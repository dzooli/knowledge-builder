FROM ollama/ollama:latest

# Pre-build custom model into an internal store, then seed it into /root/.ollama at runtime
ENV OLLAMA_HOST=0.0.0.0:11434
ENV OLLAMA_MODELS=/opt/ollama

# Ensure directories
RUN mkdir -p /opt/ollama /root/.ollama

# Add Modelfile (from this build context)
# COPY Modelfile /Modelfile

# Build the model during image build so runtime is instant
RUN /bin/sh -lc 'set -e; \
  (OLLAMA_MODELS=/opt/ollama ollama serve & pid=$!; \
   for i in $(seq 1 120); do OLLAMA_MODELS=/opt/ollama ollama list >/dev/null 2>&1 && break || sleep 1; done; \
   OLLAMA_MODELS=/opt/ollama ollama pull gpt-oss:20b || true; \
   kill $pid || true); \
  wait || true'
