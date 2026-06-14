FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Expose agent and monitor ports
EXPOSE 60000 60001

# Run simulator in headless mode (no GUI)
CMD ["python", "-m", "rcsssmj", "--host", "0.0.0.0", "--no-render"]