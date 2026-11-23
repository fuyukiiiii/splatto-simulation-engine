# The Simulation engine as a Hugging Face Docker Space (see space/README.md).
# GreenLight-Gym2 is cloned at the commit the Lab was built and tested against and installed
# unmodified, without its training extras (no RL training in the hosted engine).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

ARG GYM2_COMMIT=da8b22ca68ebb74e34cd5c7881b90c2b14e41ced
RUN git clone https://github.com/BartvLaatum/GreenLight-Gym2 vendor/GreenLight-Gym2 \
 && git -C vendor/GreenLight-Gym2 checkout --quiet "$GYM2_COMMIT"
RUN pip install --no-cache-dir --user -e vendor/GreenLight-Gym2 "fastapi==0.115.6" "uvicorn==0.34.0"

COPY --chown=user service ./service
ENV GYM2_ROOT=/home/user/app/vendor/GreenLight-Gym2
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
