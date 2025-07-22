# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code into the container
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Set the entrypoint to run the Flask app
CMD ["python", "app.py"]
