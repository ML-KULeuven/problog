FROM savente/pysdd:latest

WORKDIR /src

COPY problog_example.py /src 

RUN pip install problog 

CMD "bash"
