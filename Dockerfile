ARG PY_VERSION=3.5-stretch
ARG RDKIT_VERSION=2019.03-py35
ARG DATA_VERSION=dev

FROM registry.gitlab.com/mlpds_mit/askcos/askcos/rdkit:$RDKIT_VERSION as rdkit

FROM registry.gitlab.com/mlpds_mit/askcos/makeit-data:$DATA_VERSION as data

FROM python:$PY_VERSION

COPY --from=rdkit /usr/local/rdkit-2019-03/rdkit /usr/local/rdkit-2019-03/rdkit
COPY --from=rdkit /usr/local/rdkit-2019-03/lib /usr/local/rdkit-2019-03/lib

RUN apt-get update && \
    apt-get install -y libboost-thread-dev libboost-python-dev libboost-iostreams-dev python-tk libopenblas-dev libeigen3-dev libcairo2-dev pkg-config python-dev python-mysqldb && \
    useradd -ms /bin/bash askcos

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm requirements.txt

COPY --from=data /data /usr/local/ASKCOS/makeit/data

COPY --chown=askcos:askcos . /usr/local/ASKCOS

WORKDIR /home/askcos
USER askcos

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:"/usr/local/rdkit-2019-03/lib"
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/rdkit-2019-03:/usr/local/ASKCOS:/usr/local/ASKCOS/askcos/

RUN python /usr/local/ASKCOS/askcos/manage.py collectstatic --noinput

LABEL version={VERSION} \
      git.hash={GIT_HASH} \
      git.date={GIT_DATE} \
      git.describe={GIT_DESCRIBE}
