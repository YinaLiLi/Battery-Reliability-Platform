FROM python:3.12-slim

WORKDIR /app
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard /app/dashboard
COPY src/dashboard_data.py /app/src/dashboard_data.py
ENV PYTHONPATH=/app
EXPOSE 8501
HEALTHCHECK CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
