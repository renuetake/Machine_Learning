# SARSA

## 概要

SARSAのプログラムが格納されたディレクトリです。

強化学習ではQ値と呼ばれる、行動を決める時の指標となる数値をもっています。
SARSAでは、そのQ値を更新する時の式は以下のようになります。

<p align="center">
<img src=https://latex.codecogs.com/gif.latex?Q(s_{t},&space;a_{t})&space;\leftarrow&space;Q(s_{t},&space;a_{t})&space;&plus;&space;\alpha(r_{s_{t&plus;1}}&space;&plus;&space;\gamma&space;max_{a_{t&plus;1}}Q(s_{t&plus;1},&space;a_{t&plus;1})&space;-&space;Q(s_{t},&space;a_{t})) />
</p>
ここで、sは状態、aは行動、rは報酬、アルファは学習率、ガンマは割引率と呼ばれています。

ガンマがかかっている部分に注目して欲しいのですが、Q学習の時は次時刻に得られる最大のQ値を取っていましたが、SARSAでは次時刻の行動まで決定してから、次時刻の状態と行動でQ値を求めています。
つまり、次時刻に実際に行動した結果も現在のQ値に反映されているということなので、行動選択の仕方が学習結果に影響を与えてきます。

今回はイプシロングリーディーと呼ばれる行動選択方式を採用しているため、イプシロンの確率でランダムネスが入っています。
このランダムネスまで見越してQ値を更新していくのがSARSAとなっています。

ちなみに、状態のS、行動のA、報酬のR、次時刻の状態のS、次時刻の行動のAを使ってQ値を更新しているためSARSAと呼ばれています。

- sarsa.pyがソースコードです。  
  問題設定としては、11\*19の格子空間の迷路をエージェントに解いてもらうようになっています。
  図1の0がエージェントが自由に行き来できる空間、1がスタート地点、2がゴール地点、3が落とし穴になっています。
  落とし穴に落ちたらマイナスの報酬が与えられ、スタートに戻されるようになっています。
  さらに、毎ステップで-1の報酬を与えているので、最短ルートを学習するようになっています。
  
  <img width="303" alt="2019-01-10 2 23 01" src="https://user-images.githubusercontent.com/44384430/50917171-3b9c9e00-1480-11e9-99f7-d0f2aada0beb.png">
  
  図1. 迷路設定
  
  実行し学習が終了すると標準出力に図2のような図、同ディレクトリに「Trajectory_of_Agent-Q.png」という学習が終わった後のエージェントの軌跡をプロットした図3のようなグラフがプロットされます。

  
<img width="305" alt="2019-01-10 1 58 35" src="https://user-images.githubusercontent.com/44384430/50918108-a5b64280-1482-11e9-84fd-b5ca7ebce909.png">
  
  図2. 標準出力での学習結果出力
  
 ![trajectory_of_agent-sarsa](https://user-images.githubusercontent.com/44384430/50918125-b666b880-1482-11e9-9154-69986ac72b98.png)
 
  図3. matplotlibでの学習結果出力
