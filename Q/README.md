# Q

## 概要

Q学習のプログラムが格納されたディレクトリです。

強化学習ではQ値と呼ばれる、行動を決める時の指標となる数値をもっています。
Q学習では、そのQ値を更新する時の式は以下のようになります。

<p align="center">
<img src=https://latex.codecogs.com/gif.latex?Q(s_{t},&space;a_{t})&space;\leftarrow&space;Q(s_{t},&space;a_{t})&space;&plus;&space;\alpha(r_{s_{t&plus;1}}&space;&plus;&space;\gamma&space;max_{a_{t&plus;1}}Q(s_{t&plus;1},&space;a_{t&plus;1})&space;-&space;Q(s_{t},&space;a_{t})) />
</p>
ここで、$s$は状態、aは行動、rは報酬、アルファは学習率、ガンマは割引率と呼ばれています。

ガンマがかかっている部分が、特徴的な部分の一つで、次時刻に得られると見込まれている最大のQ値を利用して学習しています。

- q.pyがソースコードです。  
  問題設定としては、11$\times$19の格子空間の迷路をエージェントに解いてもらうようになっています。
  図1の0がエージェントが自由に行き来できる空間、1がスタート地点、2がゴール地点、3が落とし穴になっています。
  落とし穴に落ちたらマイナスの報酬が与えられ、スタートに戻されるようになっています。
  さらに、毎ステップで-1の報酬を与えているので、最短ルートを学習するようになっています。
  
  <img width="303" alt="2019-01-10 2 23 01" src="https://user-images.githubusercontent.com/44384430/50917171-3b9c9e00-1480-11e9-99f7-d0f2aada0beb.png">
  
  図1. 迷路設定
  
  実行し学習が終了すると標準出力に図2のような図、同ディレクトリに「Trajectory_of_Agent-Q.png」という学習が終わった後のエージェントの軌跡をプロットした図3のようなグラフがプロットされます。

  
  <img width="304" alt="2019-01-10 1 58 24" src="https://user-images.githubusercontent.com/44384430/50917235-5d962080-1480-11e9-83e9-39d6625494e6.png">
  
  図2. 標準出力での学習結果出力
  
 ![trajectory_of_agent-q](https://user-images.githubusercontent.com/44384430/50917759-bf0abf00-1481-11e9-8540-c8921689c507.png)
 
  図3. matplotlibでの学習結果出力
