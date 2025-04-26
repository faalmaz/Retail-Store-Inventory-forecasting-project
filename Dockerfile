FROM python:3.9-slim

WORKDIR /my_project

COPY requirements.txt .
COPY project.ipynb .
COPY app.py .
COPY retail_store_inventory.csv .

RUN pip install --requirement requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]

