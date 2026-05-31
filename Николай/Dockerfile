FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ config/
COPY models/ models/
COPY src/ src/
COPY data/alfa_onboarding_dataset_5000.csv data/

ENV PYTHONPATH=/app/src
CMD ["python3", "src/scoring.py"]
