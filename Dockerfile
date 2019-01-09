FROM python:3.6
# FROM tensorflow/tensorflow:latest-py3

# ソースを置くディレクトリを変数として格納                                                  
ARG project_dir=/work

# 必要なファイルをローカルからコンテナにコピー
RUN mkdir -p $project_dir

# Matplotlib 用の設定ファイルを用意する。
WORKDIR /etc
RUN echo "backend : Agg" >> matplotlibrc

# # Matplotlib をインストールする。
# WORKDIR /opt/app
# ENV MATPLOTLIB_VERSION 3.0.0
# RUN pip install matplotlib==$MATPLOTLIB_VERSION

# requirements.txtに記載されたパッケージをインストール                         
WORKDIR $project_dir
ADD ./requirements.txt $project_dir

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# install packages used inside container (optional)
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    && apt-get autoremove \
    && apt-get clean

