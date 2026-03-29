FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create output directory
RUN mkdir -p output_script outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose ports
# 8501 for Streamlit
# 5000 for potential API (future use)
EXPOSE 8501

# Default command: run the analysis script
# Override with `docker run -it <image> streamlit run dashboard_llm_streamlit.py` for dashboard
CMD ["python", "analyse_csat_complete_standalone.py"]
