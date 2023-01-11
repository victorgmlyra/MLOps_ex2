# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/processed data/processed
COPY Makefile Makefile

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

COPY test_environment.py test_environment.py
# COPY models/ models/
# COPY reports/ reports/

# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
ENTRYPOINT ["make", "train"]

