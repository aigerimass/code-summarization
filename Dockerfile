# Use the official Python image as a base image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the poetry-related files into the container
COPY pyproject.toml poetry.lock /app/

# Install project dependencies
RUN poetry install --no-dev

# Copy the rest of the application code
COPY . /app

# Expose port 80 for the FastAPI application
EXPOSE 80

# Command to run the FastAPI application
CMD ["poetry", "run", "start"]
