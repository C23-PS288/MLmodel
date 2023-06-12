FROM python:3.10.3-slim-buster

# Make working directories
WORKDIR  /app

# Copy every file in the source folder to the created working directory
COPY  . .

RUN pip install --upgrade pip

# Install the required packages
RUN pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

ENV HOST 0.0.0.0

EXPOSE 9001

# Run the python application
CMD ["python", "main.py"]