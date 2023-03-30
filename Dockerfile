FROM continuumio/miniconda3
RUN conda install git pip
RUN pip install git+https://github.com/antolonappan/marcia.git

