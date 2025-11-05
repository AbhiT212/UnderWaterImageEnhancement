# 1. Use an official Python base image
FROM python:3.10-slim

# --- THIS IS THE CORRECTED LINE ---
# Install the minimal headless dependency for OpenCV
RUN apt-get update && apt-get install -y libgl1

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your *entire* project code into the container
COPY . .

# 5. Expose the default Cloud Run port
EXPOSE 8080

# 6. Define healthcheck to use the $PORT variable
HEALTHCHECK CMD sh -c 'curl --fail http://localhost:$PORT/_stcore/health'

# 7. The command to run your app
CMD python -m streamlit run app.py --server.port=$PORT --server.headless=true
