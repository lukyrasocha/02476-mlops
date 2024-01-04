# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install the project (in editable mode) and its dependencies
RUN pip install . --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_02476/predict_model.py"]