# Use slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY api/ api/
COPY src/ src/
COPY data/ data/         # if you have local data for testing
COPY api/model.pkl api/scaler.pkl api/  # pre-trained models if needed

# Expose the port for FastAPI
EXPOSE 8000

# Start FastAPI app using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
