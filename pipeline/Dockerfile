FROM bentoml/model-server:0.12.0-py38

# Prevent Python writing .pyc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure log printed to stdout without buffer
ENV PYTHONUNBUFFERED 1

RUN apt-get --allow-releaseinfo-change update \
    && apt-get install -y wget \
    && apt-get install -y unzip \
    && apt-get clean;

ENV APP_DIR=/home/rec
WORKDIR $APP_DIR

# Download assignment data
RUN wget -P data/ https://cho.pe/dataset
RUN unzip data/dataset -d data/

COPY ./requirements.txt requirements.txt

COPY ./train.py train.py
COPY ./pack_service.py pack_service.py
COPY ./bento_lfm_artifacts.py bento_lfm_artifacts.py
COPY ./bento_lfm_service.py bento_lfm_service.py
COPY ./pop_rec_artifact.py pop_rec_artifact.py

# Set BentoML directory to mounted volume for shared context and data between pipeline jobs
ENV BENTOML_HOME=${APP_DIR}/data/bentoml
