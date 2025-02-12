# Use an official Python runtime as the base image
FROM python:3.8-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 5000  # Default to Flask port, can be overridden for Streamlit
ENV DATA_TYPE "dummy"  # Default to dummy data, can be set to "real" for real data

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project, ensuring all necessary files are included
COPY . .

# Create directories for artifacts, models, data, and research
RUN mkdir -p artifacts/data_ingestion artifacts/data_transformation models data research scripts

# Make port 5000 available (Flask) and 8501 (Streamlit) to the world outside this container
EXPOSE 5000
EXPOSE 8501

# Health check file for deployment verification
RUN echo "Healthy" > /app/health.txt

# Run the application based on environment variable, with data type consideration
CMD ["sh", "-c", "if [ \"$APP_TYPE\" = \"streamlit\" ]; then streamlit run streamlit_app.py --server.port 8501; else python app.py --data-type $DATA_TYPE; fi"]