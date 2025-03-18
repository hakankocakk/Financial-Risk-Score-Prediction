FROM python:3.11.4-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8001

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8001"]
