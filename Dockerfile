FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libx11-6 \
    libxcb1 \
    libglib2.0-0

# Create venv
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py"]
