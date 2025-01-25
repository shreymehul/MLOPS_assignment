# Use official Python image as a base
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to install the dependencies
COPY requirements.txt /app/

# Copy the models and app.py into the container
COPY models /app/models/
COPY app.py /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the Flask app will run on
EXPOSE 5000

# Run the Flask app when the container starts
CMD ["python", "app.py"]
