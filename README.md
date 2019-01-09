# Machine_Learning 
# 概要
これまでに実装した機械学習のプログラムが格納されたリポジトリです。

- back_propagationディレクトリ  
  ニューラルネットワークの誤差逆伝播法を実装したファイルが格納されているディレクトリです。

- RNN ディレクトリ  
  エルマンさんの「Finding Structure in Time」を実装したファイルが格納されているディレクトリです。

- Q ディレクトリ  
  強化学習のQ-学習を実装したファイルが格納されているディレクトリです。

- SARSA ディレクトリ  
  強化学習のSARSAを実装したファイルが格納されているディレクトリです。

全てDockerで環境を統一しているので、以下の通りにイメージファイル・コンテナを作成してからDocker上で実行すれば再現できるはずです。


## Dockerの使い方

Dockerで環境を構築する
```
# イメージ作成(Dockerfileがあるディレクトリで実行)
$ docker build ./ -t machine_learning

# コンテナ作成
$ docker run -v $PWD:/work -itd --name machine_learning_01 machine_learning
```

コンテナの開始/停止
```
# コンテナの停止
$ docker stop machine_learning_01

# コンテナの開始
$ docker start machine_learning_01
```

Dockerで実行を行う
```
# python実行
$ docker exec -ti machine_learning_01 python <filename.py>

# bashの実行
$ docker exec -ti machine_learning_01 bash
```


## Install Test

パッケージが正しくインストールされているかのテスト
```
$ docker exec -ti machine_learning_01 bash
$ cd test
$ python installtest.py
```
