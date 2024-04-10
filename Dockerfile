FROM python:3.9

WORKDIR /usr/src/app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.25.2 matplotlib==3.8.2 tensorflow==2.16.1 jupyter==1.0.0

COPY . .

EXPOSE 8888

# notebook --no-browser --allow-root --ip=0.0.0.0 --port=8888
