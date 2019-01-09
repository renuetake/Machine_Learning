import math
import numpy as np
import matplotlib.pyplot as plt
import os

# パスの設定
DIR = os.getcwd()
if (DIR.split('/')[-1] != 'RNN'):
    DIR = DIR + '/RNN'
def init_data(data_len):
    """
    入力するデータを生成する
    input:
        data_len: int 入力するデータの長さ(3の倍数)
    output:
        res_data: リスト 入力するデータ(0と1のビット列)
    """

    res_data = []
    for i in range(data_len):
        # 最初の1ビット目の処理
        if (i == 0):
            res_data.append(np.random.randint(2))
        # それ以降のビットの処理
        elif ((i+1) % 3 != 0):
            res_data.append(np.random.randint(2))
        # 3の倍数ビットの時は、前と前々の2ビットのXORが入る
        else:
            res_data.append(int(res_data[i-1] ^ res_data[i-2]) )
    
    return res_data



def init_net(num_input, num_context, num_hidden, num_output):
    """
    ニューラルネットワークの重みの初期化
    入力層・隠れ層・出力層のユニット数が入力されると、
    初期化された重み(numpy配列)のリストを返す。
    input:
        num_input :  int 入力層のユニット数
        num_hidden: int 出力層のユニット数
        num_output: int 出力層のユニット数
    output:
        network: [np.aray, np.array] 入力層~隠れ層、隠れ層~出力層それぞれの重み
    """
    # 引数がint以外だったらValueErrorを投げる
    if ((type(num_input) != int) or (type(num_hidden) != int) or (type(num_output) != int)):
        raise ValueError


    # 入力層~隠れ層の重みを(-0.5~0.5)の範囲で初期化
    weight1 = np.random.rand(num_input+num_context+1, num_hidden) - 0.5
    weight2 = np.random.rand(num_hidden+1, num_output) - 0.5
    return [weight1, weight2]

def feedforward(weight, data, isample, context, beta=0.8):
    """
    重みのリストを入力すると、各層の出力値がリストで返す。
    input:
        weight: リスト [np.array, np.array]
        data: np.array 入力値と出力値のデータが入ったnp.array
        isample: int サンプル番号
        context: リスト 前の時刻の隠れ層の出力値
        beta: float シグモイド関数に使われる定数(初期値0.8)
    output:
        result: [X[num_input], H[num_hidden], Y[num_output]]
            X : 入力層の出力値
            H : 隠れ層の出力値
            Y : 出力層の出力値
    """
    
    # 引数のweightがリスト、isampleがint以外だったらValueErrorを投げる
    if ((type(weight) != list) or (type(data) != list) or (type(isample) != int)):
        print("weight:{}\tdata:{}\tisample:{}".format(type(weight), type(data), type(isample)))
        raise ValueError
    if (weight[0] is int):
        raise ValueError

    X = []
    H = []
    Y = []

    # 入力層のユニット数だけ繰り返し
    for i in range(NUM_INPUT):
        # 入力層のそれぞれのユニットの出力は読み込んだデータと同じ
        X.append(data[isample])
    # 前時刻の隠れ層の出力をコンテキスト層からの出力にする
    for i in range(NUM_CONTEXT):
        X.append(context[i])
    # 閾値用にX[NUM_INPUT] = 1.0にする
    X.append(1.0)
    
    # 隠れ層のユニット数だけ繰り返し
    for i in range(NUM_HIDDEN):
        # 隠れ層のそれぞれのユニットの出力は、
        # そのユニットに入力された値の総和をシグモイド関数に入れて計算される
        net_input = 0
        # 入力層のそれぞれのユニットから入力された値を加算
        for j in range(NUM_INPUT+NUM_CONTEXT+1):
            net_input = net_input + (weight[0][j][i] * X[j])
        # 加算された値をシグモイド関数に適用
        H.append((1.0 / (1.0 + np.exp(net_input * -beta))))
    
    H.append(1.0)
    # 出力層のユニット数だけ繰り返し
    for i in range(NUM_OUTPUT):
        net_input = 0
        for j in range(NUM_HIDDEN+1):
            net_input = net_input + (weight[1][j][i] * H[j])
        Y.append((1.0 / (1.0 + np.exp(net_input * -beta))))
    return [X, H, Y]

def backward(weight, data, isample, out, beta=0.8):
    """
    逆方向に伝播させて、教師データとの差を計算する
    input:
        weight: リスト [np.array, np.array]
        data: np.array 入力値と出力値のデータが入ったnp.array
        isample: int サンプル番号
        beta: float シグモイド関数に使われる定数(初期値0.8)
    output:
        back: [H[num_hidden], Y[num_output]] 出力値から隠れ層・出力層のそれぞれのユニットからとった差を格納したリスト
    """
    if ((type(weight) != list) or (type(data) != list) or (type(isample) != int) or (type(out) != list)):
        raise ValueError
    H = []
    Y = []
    # 出力層から逆伝播させる
    for i in range(NUM_OUTPUT):
        # 3000ビット目の出力の正否判定は0ビット目の値を使う
        if (isample+1 >= 3000):
            Y.append(beta * (data[0] - out[2][i]) * (1.0 - out[2][i]) * out[2][i])
        else:
            Y.append(beta * (data[isample+1] - out[2][i]) * (1.0 - out[2][i]) * out[2][i])
    
    # 隠れ層から逆伝播させる
    for i in range(NUM_HIDDEN):
        net_input = 0
        for j in range(NUM_OUTPUT):
            net_input = net_input + (weight[1][i][j] * Y[j])
        H.append(beta * net_input * (1.0 - out[1][i]) * out[1][i])
    return [H, Y]

def modify_weights(weight, out, back, epsilon=0.05):
    """
    逆伝播の結果を用いて重みを修正する
    input:
        weight: リスト 各層・各ユニットの重みが格納されたリスト
        back: リスト backward()から返ってきたリスト
        out: リスト　feedforward()から返ってきたリスト
        epsilon: float 学習率。初期値0.05
    output:
        なし
    """
    # 入力された型が正常か判定
    if ((type(weight) != list) or (type(back) != list) or (type(epsilon) != float)):
        raise ValueError
    for i in range(NUM_INPUT+NUM_CONTEXT+1):
        for j in range(NUM_HIDDEN):
            weight[0][i][j] = weight[0][i][j] + epsilon * out[0][i] * back[0][j]
    for i in range(NUM_HIDDEN+1):
        for j in range(NUM_OUTPUT):
            weight[1][i][j] = weight[1][i][j] + epsilon * out[1][i] * back[1][j]

def print_results(isample, out, data, error):
    """
    学習結果をプリントする
    input:
        isample: int サンプル番号
        out: リスト feedforward()から出力されたリスト
        data: リスト 入力値と出力値が格納されたリスト
        error: float エラー値
    output:
        なし
    """
    print('\t training data No. = {}'.format(isample+1))
    print('\t \t IN: {}'.format(data[isample]))
    print('\t Trained_OUT: ', end="")
    if (isample+1 >= 3000):
        print('{}'.format(data[0]))
    else:
        print('{}'.format(data[isample+1]))
    print('\t \t OUT: {}'.format(out[2]))
    print('\t error = {}'.format(error))
    print("")

def calc_error(isample, data, out):
    """
    出力値からエラー値を計算する
    input:
        isample: int サンプル番号
        data: リスト 入力値と出力値の教師データが格納されたリスト
        out: feedforward()が出力したリスト
    output:
        error: float エラー値
    """
    error = 0.0
    for i in range(NUM_OUTPUT):
        error += (data[isample] - out[2][i]) ** 2

    error /= 2.0

    return error

def plot_error(file_name='RNN_error.dat'):
    """
    エラー値の推移をプロットする
    input:
        error: リスト 毎epochのエラー値が格納されたリスト
    output:
        なし
    """
    error = []
    with open(DIR+'/'+file_name, 'r') as f:
        for ele in f:
            error.append(float(ele))
    # 縦軸・横軸のラベル設定
    plt.xlabel('cycle')
    plt.ylabel('error')

    # 3000ビット分表示させると潰れて見えないため
    # 0-10ビット分を表示させる
    plt.xlim([0, 13])

    # プロット
    plt.plot(error, label="error_value")

    plt.xticks(np.arange(0, 13, 1))
    # 凡例の表示
    plt.legend()

    plt.grid()

    # pngファイルに書き出し
    plt.savefig(DIR+'/'+'error_cycle.png')


if __name__ == '__main__':
    NUM_LEARN = 1200           # 学習の繰り返し回数
    LEN_DATA = 3000
    global NUM_INPUT               # 入力層のユニット数
    NUM_INPUT = 1
    global NUM_HIDDEN              # 隠れ層のユニット数
    NUM_HIDDEN = 3
    global NUM_CONTEXT              # コンテキスト層のユニット数
    NUM_CONTEXT = NUM_HIDDEN
    global NUM_OUTPUT              # 出力層のユニット数
    NUM_OUTPUT = 1
    THRESHOLD_ERROR = 0.001     # 学習誤差がこの値以下になるとプログラムが停止する
    error_list = []           # 1epoch中のエラー値の推移(参考文献ではこの値が周期的になっていた)
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # 乱数シード
    '''
    # シードの数だけループ
    for seed in seeds:
        error = []
        print("seed {} is start!".format(seed))
        # 乱数シードの値を設定する
        np.random.seed(seed=seed)
        
        # 入出力データの読み込み
        data = init_data(LEN_DATA)

        # ネットワークの重みの初期化
        weight = init_net(NUM_INPUT, NUM_CONTEXT, NUM_HIDDEN, NUM_OUTPUT)

        # 1epoch目のコンテキスト層の初期化
        context = np.random.rand(NUM_CONTEXT) / 10

        # 学習の繰り返しループ
        for ilearn in range(NUM_LEARN):
            error_list = []
            if (ilearn % 100) == 0:
                print('# of learning : {}'.format(ilearn))
            
            # 訓練データに関するループ
            for isample in range(LEN_DATA):
                out = feedforward(weight, data, isample, context)
                context = out[1] # 次の時刻の隠れ層に入力されるコンテキスト層の設定
                if (ilearn == (NUM_LEARN-1)):
                    error.append(calc_error(isample, data, out))            
                back = backward(weight, data, isample, out)
                modify_weights(weight, out, back, epsilon=0.1)
        
        error_list.append(error)   
        print("seed {} is done!".format(seed))

    sum_error = np.zeros(3000)
    for error in error_list:
        sum_error += np.array(error)  
    ave_error = sum_error / len(seeds)

    # 出力されたエラー値の平均を外部ファイルに書き出し
    with open(DIR+'/RNN_error.dat', mode='w') as f:
        for ele in ave_error:
            f.write(str(ele)+'\n')
    '''
    plot_error()

