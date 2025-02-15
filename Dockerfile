FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN apt-get update && apt-get install -y build-essential # Required for prettier

# Install Node.js and npm
RUN apt-get update && apt-get install -y nodejs npm
RUN npm install --global prettier@3.4.2

EXPOSE 8000

CMD ["uvicorn", "TDS-P1:app", "--host", "0.0.0.0", "--port", "8000"]