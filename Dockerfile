# Use an official Python image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the dependency files first for better caching
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Copy the rest of the application files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["poetry", "run", "streamlit", "run", "csv_agent/app.py"]
