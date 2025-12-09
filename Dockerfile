# Dockerfile

# Use the precise Python version (adjusting from 3.11.9 to the latest available patch if necessary)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# Set the working directory
WORKDIR /app

# Copy requirements file first (for optimal Docker layer caching)
COPY requirements.txt .

# Install dependencies (This is the step that must pass)
# --no-cache-dir saves space and prevents local cache issues
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE $PORT

# Define the command to run your application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]