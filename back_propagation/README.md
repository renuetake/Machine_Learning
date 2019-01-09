# Back_Propagation

## 概要

ニューラルネットワークのバックプロパゲーションを実装したプログラムが格納されたディレクトリです。

- 「bp.py」がソースコードです。

- 「data.dat」はニューラルネットワークに期待する入出力が記述されたファイルです。

以下のXOR問題が記述されています。形式は「入力 入力 出力」となっています。



0 0 1

0 1 0

1 0 0

1 1 1

- 「bp_test.py」は「bp.py」のエラー値をプロットするように改良したものです。
実行すると「errpr_per_epoch.pdf」のようなグラフがプロットされます。

![error_per_epoch](https://user-images.githubusercontent.com/44384430/50677018-df2e1e80-103a-11e9-9e0b-7e4dc45c81ed.jpg)
