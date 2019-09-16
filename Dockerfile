FROM python:3

WORKDIR /src

COPY problog_example.py /src 

USER root

RUN apt-get update && apt-get install gcc git  && \
	git clone https://github.com/wannesm/PySDD && \
	python -m pip install cython cysignals numpy problog

COPY sdd-2.0 /src/sdd-2.0

COPY libsdd-2.0 /src/libsdd-2.0

RUN	cp -r /src/libsdd-2.0/ /src/PySDD/pysdd/lib/libsdd-2.0  && \
	cp -r /src/sdd-2.0/ /src/PySDD/pysdd/lib/sdd-2.0  && \
	cd PySDD && python setup.py install 

CMD "bash"
#CMD ["python", "problog_example.py"]
