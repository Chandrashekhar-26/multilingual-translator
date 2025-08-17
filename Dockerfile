# use existing Python image
FROM python:3.10-slim

# To avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
#WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies from requirements
RUN pip install -r requirements.txt

# copy complete code
COPY . .

# Expose the port
EXPOSE 5001

# Run Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5001"]
