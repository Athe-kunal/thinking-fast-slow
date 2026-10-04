# Sandbox for executing model-written code (HumanEval / MBPP / KodCode tests).
# Runs with --network none; only the standard library and pytest are present.
#   docker build -t thinking-fast-slow-sandbox -f docker/sandbox.Dockerfile docker
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.4
