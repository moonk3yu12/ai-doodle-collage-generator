FROM python:3.11-slim

# HF Spaces Docker requires a non-root user with uid=1000
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install dependencies before copying app code (better layer caching)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=user:user . .

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "app.py"]
