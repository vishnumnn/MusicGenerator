FROM continuumio/miniconda3
MAINTAINER Vishnu Menon

WORKDIR /home/vishnum/source/MusicGenerator

COPY . .

RUN conda install --file requirements.txt

CMD ["python", "./src/main.py"]
