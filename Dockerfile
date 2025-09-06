# Use the standard python image instead of slim, as it has more build tools
FROM python:3.11

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file first (better for Docker layer caching)
COPY requirements.txt .

# 1. Install system build dependencies required to compile Python packages
# 2. Install pip packages
# 3. Remove the build dependencies and clean up apt cache in the SAME LAYER to keep image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y gcc g++ \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of your application's code into the container at /app
COPY . .

# Expose port 8501 for the Streamlit app
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]