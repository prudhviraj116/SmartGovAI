# ==========================
# SmartGovAI Cloud Run Dockerfile
# ==========================
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit specific setup
EXPOSE 8080

# Cloud Run expects PORT environment variable
ENV PORT 8080

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
