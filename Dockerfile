FROM python:3.11.4-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgomp1
RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
