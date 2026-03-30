FROM python:3.11-slim AS builder

WORKDIR /app

# Create venv
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
