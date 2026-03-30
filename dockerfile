# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Train model artifacts at build time so startup is instant
RUN python train.py

# Render (and most cloud platforms) inject PORT at runtime
EXPOSE 5000
ENV FLASK_APP=run.py
ENV PORT=5000

# Shell form so $PORT is expanded at runtime
CMD gunicorn -b 0.0.0.0:$PORT run:app
