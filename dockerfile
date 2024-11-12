# Use an official lightweight Python base image
FROM python:3.9-slim


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


CMD ["python", "Q4.py"]
