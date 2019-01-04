import math
import numpy as np


def read_data(path='data.dat'):
    """
    入力データと教師データが格納されている「data.dat」を開き、
    それぞれのデータをnumpy配列に格納して返す。
        data.datの書式
            データ,データ,データ
    input:
        なし
    output:
        data: np.array
    """
    try:
        data = np.loadtxt(path, delimiter=' ')
    except OSError:
        print('読み込むファイル名を確認してください')
        raise OSError
    return data

def init_net(num_input, num_hidden, num_output, seed=1):
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
        # print('num_input:{}\tnum_hidden:{}\tnum_output:{}'.format(type(num_input), type(num_hidden), type(num_output)))
        raise ValueError

    # 乱数シードの値を設定する
    np.random.seed(seed=seed)


    # 入力層~隠れ層の重みを(-0.5~0.5)の範囲で初期化
    weight1 = np.random.rand(num_input+1, num_hidden) - 0.5
    weight2 = np.random.rand(num_hidden+1, num_output) - 0.5
    """test"""
    # weight1 = np.ones((num_input+1, num_hidden))
    # weight2 = np.ones((num_hidden+1, num_output))
    """ここまで"""
    return [weight1, weight2]

def feedforward(weight, data, isample, beta=0.8):
    """
    重みのリストを入力すると、各層の出力値がリストで返す。
    input:
        weight: リスト [np.array, np.array]
        data: np.array 入力値と出力値のデータが入ったnp.array
        isample: int サンプル番号
        beta: float シグモイド関数に使われる定数(初期値0.8)
    output:
        result: [X[num_input], H[num_hidden], Y[num_output]]
            X : 入力層の出力値
            H : 隠れ層の出力値
            Y : 出力層の出力値
    """
    
    # 引数のweightがリスト、isampleがint以外だったらValueErrorを投げる
    if ((type(weight) != list) or (type(data) != np.ndarray) or (type(isample) != int)):
        # print('weight:{}\tdata:{}\tisample:{}'.format(type(weight), type(data), type(isample)))
        raise ValueError
    if (weight[0] is int):
        # print('入力された重みが不正です')
        raise ValueError

    X = []
    H = []
    Y = []
    # 入力層のユニット数だけ繰り返し
    for i in range(NUM_INPUT):
        # 入力層のそれぞれのユニットの出力は読み込んだデータと同じ
        X.append(data[isample][i])
    # 閾値用にX[NUM_INPUT] = 1.0にする
    X.append(1.0)
    # 隠れ層のユニット数だけ繰り返し
    for i in range(NUM_HIDDEN):
        # 隠れ層のそれぞれのユニットの出力は、
        # そのユニットに入力された値の総和をシグモイド関数に入れて計算される
        net_input = 0
        # 入力層のそれぞれのユニットから入力された値を加算
        for j in range(NUM_INPUT+1):
            net_input = net_input + (weight[0][j][i] * X[j])
        # 加算された値をシグモイド関数に適用
        # print("H[{}] = 1/1+exp({} * -{}) = {}".format(i, net_input, beta, (1/(1+np.exp(net_input * -beta)))))
        H.append((1.0 / (1.0 + np.exp(net_input * -beta))))
    H.append(1.0)
    # 出力層のユニット数だけ繰り返し
    for i in range(NUM_OUTPUT):
        # 出力層のそれぞれのユニットの出力は、
        # そのユニットに入力された値の総和をシグモイド関数に入れて計算される
        net_input = 0
        for j in range(NUM_HIDDEN+1):
            net_input = net_input + (weight[1][j][i] * H[j])
        # print("Y[{}] = 1/1+exp({} * -{}) = {}".format(i, net_input, beta, (1/(1+np.exp(net_input * -beta)))))
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
    if ((type(weight) != list) or (type(data) != np.ndarray) or (type(isample) != int) or (type(out) != list)):
        # print('weight:{}\tdata:{}\tisample:{}'.format(type(weight), type(data), type(isample)))
        raise ValueError
    H = []
    Y = []
    # 出力層から逆伝播させる
    for i in range(NUM_OUTPUT):
        # print("{} * (data[{}][{}] - out[2][{}]) * (1.0 - out[2][{}]) * out[2][{}]".format(beta, isample, NUM_INPUT+1, i, i, i))
        # print("{} * ({} - {}) * (1.0 - {}) * {}".format(beta, data[isample][NUM_INPUT+i], out[2][i], out[2][i], out[2][i]))
        # print("Y_back[{}] = {} * ({} - {}) * (1.0 - {}) * {} = {}".format(i, beta, data[isample][NUM_INPUT+i], out[2][i], out[2][i], out[2][i], (beta * (data[isample][NUM_INPUT+i] - out[2][i]) * (1.0 - out[2][i]) * out[2][i])))
        Y.append(beta * (data[isample][NUM_INPUT+i] - out[2][i]) * (1.0 - out[2][i]) * out[2][i])
    
    # 隠れ層から逆伝播させる
    for i in range(NUM_HIDDEN):
        net_input = 0
        for j in range(NUM_OUTPUT):
            net_input = net_input + (weight[1][i][j] * Y[j])
        # print("H_back[{}] = {} * ({} - {}) * (1.0 - {}) * {} = {}".format(i, beta, net_input, out[1][i], out[1][i], out[1][i], (beta * net_input * (1.0 - out[1][i]) * out[1][i])))
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
    # print("INPUT_OUT = {}".format(out))
    for i in range(NUM_INPUT+1):
        for j in range(NUM_HIDDEN):
            # print("before weight[0][{}][{}] : {}".format(i, j, weight[0][i][j]))
            # print("weight[0][{}][{}] = {} + {} * {} * {} = {}".format(i, j, weight[0][i][j], epsilon, out[0][j], back[0][j], (weight[0][i][j] + epsilon * out[0][j] * back[0][j])))
            weight[0][i][j] = weight[0][i][j] + epsilon * out[0][i] * back[0][j]
            # print("after weight[0][{}][{}] : {}".format(i, j, weight[0][i][j]))
    for i in range(NUM_HIDDEN+1):
        for j in range(NUM_OUTPUT):
            # print("before weight[1][{}][{}] : {}".format(i, j, weight[1][i][j]))
            # print("weight[1][{}][{}] = {} + {} * {} * {} = {}".format(i, j, weight[1][i][j], epsilon, out[1][j], back[1][j], (weight[1][i][j] + epsilon * out[1][j] * back[1][j])))
            weight[1][i][j] = weight[1][i][j] + epsilon * out[1][i] * back[1][j]
            # print("after weight[1][{}][{}] : {}".format(i, j, weight[1][i][j]))
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
    print('\t \t IN: {}'.format(data[isample][:NUM_INPUT]))
    print('\t Trained_OUT: ', end="")
    print('{}'.format(data[isample][NUM_INPUT:]))
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
        error += (data[isample][(NUM_INPUT)+i] - out[2][i]) ** 2

    error /= 2.0

    return error

if __name__ == '__main__':
    NUM_LEARN = 50000           # 学習の繰り返し回数
    NUM_SAMPLE = 4              # サンプル数
    global NUM_INPUT               # 入力層のユニット数
    NUM_INPUT = 2
    global NUM_HIDDEN              # 隠れ層のユニット数
    NUM_HIDDEN = 3
    global NUM_OUTPUT              # 出力層のユニット数
    NUM_OUTPUT = 1
    THRESHOLD_ERROR = 0.001     # 学習誤差がこの値以下になるとプログラムが停止する

    # 入出力データの読み込み
    data = read_data()
    # print(data)
    # ネットワークの重みの初期化
    weight = init_net(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)
    # print(weight)
    # 学習の繰り返しループ
    for ilearn in range(NUM_LEARN):
        if (ilearn % 1000) == 0:
            print('# of learning : {}'.format(ilearn))
        
        # 訓練データに関するループ
        error = 0.0
        for isample in range(NUM_SAMPLE):
            out = feedforward(weight, data, isample)
            # print("OUT = {}".format(out))
            # print("OUT[0] = {}".format(out[0]))
            # print("OUT[1] = {}".format(out[1]))
            # print("OUT[2] = {}".format(out[2]))
            error += calc_error(isample, data, out)

            if (ilearn % 1000) == 0:
                print_results(isample, out, data, error)
            
            back = backward(weight, data, isample, out)
            # print("BACK = {}".format(back))
            modify_weights(weight, out, back, epsilon=0.2)
        
        if (error < THRESHOLD_ERROR):
            break
    
    print("\n\n# of learning : {}\n".format(ilearn))
    for i in range(NUM_SAMPLE):
        out = feedforward(weight, data, i)
        print_results(i, out, data, calc_error(i, data, out))
