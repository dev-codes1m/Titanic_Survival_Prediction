FROM python:3.10-slim-buster
ENV PYTHONDONTWRITECODE 1
ENV PYTHONBUFFERED 1
RUN mkdir /webapp
WORKDIR /webapp
RUN pip install --upgrade pip
COPY requirements.txt /webapp/

RUN pip install -r requirements.txt
COPY webapp /webapp/
EXPOSE 80

CMD ["python","./webapp/manage.py","runserver","0.0.0.0:8000"]










