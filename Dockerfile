FROM apache/airflow:2.6.1

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

WORKDIR /app

COPY requirements.txt ./

USER airflow
RUN pip install --no-cache-dir -r requirements.txt

USER root
COPY . .

USER airflow