# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── HF Spaces requires uid=1000 ────────────────────────────────────────────────
RUN useradd -m -u 1000 user

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /home/user/app

# ── Install dependencies (as root for write access, then hand off) ─────────────
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY --chown=user:user . .

# ── Switch to non-root user ────────────────────────────────────────────────────
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Gradio / HF Spaces default port ───────────────────────────────────────────
EXPOSE 7860

CMD ["python", "app.py"]
