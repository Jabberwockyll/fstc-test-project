FROM python:3.9

WORKDIR /usr/src/fstc-app

COPY requirements.txt setup.py ./
COPY fewshot/ ./fewshot

RUN pip3 install -r requirements.txt
COPY ./ ./

# RUN dvc repro
# RUN dvc metrics show
# RUN dvc plots show results/confusion_plot.csv --template confusion -x actual -y predicted