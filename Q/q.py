import math
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

# パスの設定
DIR = os.getcwd()
if (DIR.split('/')[-1] != 'Q'):
    DIR = DIR + '/Q'

class Agent():
    def __init__(self, start_pos):
        """
        Agentクラスのコンストラクタ
        """
        self.start_pos = start_pos           # 開始位置
        self.current_pos = self.start_pos    # 今の位置
        self.next_pos = [None, None]         # 次時刻の位置
        self.current_action = None           # 今の行動
        self.next_action = None              # 次時刻の行動
    
    def decide_action(self, pos, qtable, epsilon):
        """
        エージェントの行動を決定
        input:
            qtable: np.ndarray Qテーブル
            epsilon: 行動決定の際のランダムネスを決める定数
        output:
            なし
        """

        ac_num = qtable.shape[2]
        x, y = pos
        ret_action = None
        # 最大のQ値を求める
        qmax = qtable[x, y].max()

        # Q値が最大の行動を決定
        # qmax_action = np.argmax(qtable[self.current_pos[0], self.current_pos[1]])
        qmax_action = [i for i in range(ac_num) if qtable[x, y, i] == qmax]
        rand = np.random.rand()
 
        # グリーディーに行動を選択することが決まった場合
        if ((1 - epsilon) > rand):
            # 今の状態でQ値が同じ行動があるか判定
            if (len(qmax_action) > 1):
                # 最大のQ値が複数あれば、その中からランダムで1つ選ぶ
                ret_action = np.random.choice(qmax_action)
            else:
                # 最大のQ値が複数無ければ、最大のQ値の行動が返される
                ret_action = qmax_action[0]
 
        # イプシロンの確率でランダムに行動が決まる
        else:
            ret_action = np.random.choice([0, 1, 2, 3])

        return ret_action
    
    def move_agent(self, row, col):
        """
        エージェントの次時刻の位置を決める
        input:
            row: int 迷路の行方向の大きさ
            col: int 迷路の列方向の大きさ
        output:
            なし
        """
        UP, RIGHT, DOWN, LEFT = (0, 1, 2, 3)
        current_x, current_y = self.current_pos
        # もし行動が上への移動の場合
        if (self.current_action == UP):
            # xが0の場所は上へ移動しても壁なので、移動させない
            if (current_x == 0):
                self.next_pos = copy.copy(self.current_pos)
            else:
                self.next_pos = [current_x-1, current_y]
        # もし行動が右への移動の場合
        elif (self.current_action == RIGHT):
            # yがcol-1の場所は右へ移動しても壁なので、移動させない
            if (current_y == col-1):
                self.next_pos = copy.copy(self.current_pos)
            else:
                self.next_pos = [current_x, current_y+1]
        # もし行動が下へ移動の場合
        elif (self.current_action == DOWN):
            # xがrow-1の場所は下へ移動しても壁なので、移動させない
            if (current_x == row-1):
                self.next_pos = copy.copy(self.current_pos)
            else:
                self.next_pos = [current_x+1, current_y]
        # もし行動が左へ移動の場合
        elif (self.current_action == LEFT):
            # yが0の場所は左へ移動しても壁なので、移動させない
            if (current_y == 0):
                self.next_pos = copy.copy(self.current_pos)
            else:
                self.next_pos = [current_x, current_y-1]
    
    def cal_reward(self, maze, row, col):
        """
        エージェントが得られる報酬を算出
        input:
            maze: np.ndarray 迷路の情報が入った配列(0:道 1:スタート 2:ゴール 3:落とし穴)
            row: int 迷路の行方向の大きさ
            col: int 迷路の列方向の大きさ
        output:
            ret_reward: int エージェントが得られる報酬
        """
        WAY, START, GOAL, HOLE = (0, 1, 2, 3)
        ret_reward = 0
        UP, RIGHT, DOWN, LEFT = (0, 1, 2, 3)
        current_x, current_y = self.current_pos
        next_x, next_y = self.next_pos
        # もし行動が上への移動で移動先が壁だった場合
        if (self.current_action == UP) & (current_x == 0):
            ret_reward += -100
        # もし行動が右への移動で移動先が壁だった場合
        elif (self.current_action == RIGHT) & (current_y == col-1):
            ret_reward += -100
        # もし行動が下への移動で移動先が壁だった場合
        elif (self.current_action == DOWN) & (current_x == row-1):
            ret_reward += -100
        # もし行動が左への移動で移動先が壁だった場合
        elif (self.current_action == LEFT) & (current_y == 0):
            ret_reward += -100
        
        # もし落とし穴に落ちていた場合
        if (maze[next_x, next_y] == HOLE):
            ret_reward += -100

        # ゴールに辿り着いたら正の報酬
        if (maze[next_x, next_y] == GOAL):
            ret_reward += 1000

        # 最短ルートを学習して欲しいため、毎Stepで負の報酬を与える
        ret_reward += -1

        return ret_reward
    
    def move_state(self, maze):
        """
        状態を遷移させる
        input:
            maze: np.ndarray  迷路の情報が格納された配列
        output:
            なし
        """
        next_x, next_y = self.next_pos
        WAY, START, GOAL, HOLE = (0, 1, 2, 3)
        start_pos = [np.where(maze == START)[0][0], np.where(maze == START)[1][0] ]
        # 穴に落ちたらスタート地点に移動させる
        if (maze[next_x, next_y] == HOLE):
            self.next_pos = copy.copy(start_pos)
        
        self.current_pos = copy.copy(self.next_pos)
        self.next_pos = [None, None]
        self.current_action = copy.copy(self.next_action)
        self.next_action = None


            

    
    

def init_map(row, col, start_pos, goal_pos):
    """
    エージェントに解かせる迷路の初期化。
    0が道、1がスタート、2がゴール、3が落とし穴
    input:
        row: int 迷路の行方向の大きさ。
        col: int 迷路の列方向の大きさ。
        start_pos: リスト スタートの位置
        goal_pos: リスト ゴールの位置
    output:
        ret_map: リスト 上記の0~3の整数が入ったリスト
    """
    WAY, START, GOAL, HOLL = (0, 1, 2, 3)
    # スタート地点のx、y
    s_x, s_y = start_pos
    # ゴール地点のx、y
    g_x, g_y = goal_pos

    # 最初は全て道として初期化
    ret_map = np.zeros((row, col))

    # スタート地点とゴール地点の設定
    ret_map[s_x, s_y] = START
    ret_map[g_x, g_y] = GOAL

    # 落とし穴を設定
    ret_map[row-1, 1:col-1] = HOLL

    return ret_map
    
def init_qtable(row, col, ac_num=4):
    """
    Qテーブルの初期化
    input:
        row: int 迷路の行方向の大きさ。
        col: int 迷路の列方向の大きさ。
        ac_num: int エージェントのアクション数。初期値4
    output:
        ret_q: np.ndarray Qテーブル
    """
    ret_q = np.random.rand(row, col, ac_num)

    return ret_q

def update_q(q_table, agent, reward, alpha, gamma):
    """
    Q値の更新
    input:
        q_table: np.ndarray Qテーブル
        agent: Agentクラスのオブジェクト
        reward: int current_posでcurrent_actionを取った時に得られた報酬
        alpha: float Q値を更新する際の幅。学習率。
        gamma: gloat 未来のQ値をどれだけ影響させるかを決める定数。割引率。
    output:
        ret_q: float 更新後のQ値
    """
    now_x, now_y = agent.current_pos # 現時刻の位置
    now_action = agent.current_action # 現時刻の行動
    now_q = q_table[now_x, now_y, now_action] # 現時刻の位置で現時刻の行動を取った時のQ値
    next_x, next_y = agent.next_pos # 次時刻の位置
    next_action = agent.next_action # 次時刻の行動
    next_max_q = q_table[next_x, next_y].max() # 次時刻の位置で取ることのできる最大のQ値
    ret_q = now_q + alpha * (reward + (gamma * next_max_q) - now_q)
    return ret_q


def show_map(maze):
    """
    迷路を視覚化(0:道 1:スタート 2:ゴール 3:落とし穴)
    input:
        map: np.ndarray 迷路の情報が入った配列
    output:
        なし
    """
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            print('{} '.format(int(maze[i, j])), end='')
        print('')
            
def plot_result(row, col, log):
    """
    結果をプロットする
    input:
        row: int 迷路の行方向の大きさ
        col: int 迷路の列方向の大きさ
        log: リスト エージェントが行動した履歴
    output:
        なし
    """
    # グラフのタイトルを設定
    plt.title('Trajectory of Agent')

    # x軸とy軸のリミットを設定
    plt.xlim([-0.5, col-0.5])
    plt.ylim([row-0.5, -0.5])

    # 散布図で描画
    plt.scatter(log[:, 1], log[:, 0], label='Trajectory', marker='s', s=50)

    # 凡例の表示
    plt.legend()

    # 目盛りを消去
    plt.tick_params(length=0)
    
    # マスの真ん中に点が描画されるように0.5刻みでグリッドを設定
    plt.yticks(np.arange(0.5, 11.5, 1), color='None')
    plt.xticks(np.arange(0.5, 19.5, 1), color='None')

    # グリッドの描画
    plt.grid()

    # pngファイルに書き出し
    plt.savefig(DIR+'/'+'Trajectory_of_Agent-Q.png')

    # 標準出力に最大のQ値を取る行動をプリント
    q_table = np.load(DIR+'/Q_table.npy')
    for i in range(11):
        for j in range(19):
            action = np.argmax(q_table[i, j])
            if (action == 0):
                print("↑ ", end='')
            elif (action == 1):
                print("→ ", end='')
            elif (action == 2):
                print('↓ ', end='')
            elif (action == 3):
                print('← ', end='')
        print("")
    


if __name__ == '__main__':
    seed = 1                    # 乱数シード
    NUM_EPISODE = 10000 # 最大エピソード
    NUM_STEP = 1000             # 最大ステップ
    EPSILON = 0.7               # 行動決定の際のランダムネスを決定する定数
    ALPHA = 0.1                 # 学習率
    GAMMA = 0.9                 # 割引率
    ROW = 11                    # 迷路の行方向の大きさ
    COL = 19                    # 迷路の列方向の大きさ
    START_POS = [ROW-1, 0]      # スタート地点
    GOAL_POS = [ROW-1, COL-1]   # ゴール地点

    # 乱数シード設定
    np.random.seed(seed=seed)

    # 迷路の初期化
    maze = init_map(ROW, COL, start_pos=START_POS, goal_pos=GOAL_POS)
    # Qテーブルの初期化
    q_table = init_qtable(ROW, COL)
    # print("Qtable.shape:{}".format(q_table.shape))

    for iepisode in range(NUM_EPISODE):
        if (iepisode % 1000) == 0:
            print('# of episode : {}'.format(iepisode))
        # エージェントの初期化
        agent = Agent(start_pos=START_POS)
        # print("current_pos:{}".format(agent.current_pos))
        # 現時刻の状態から現時刻の行動の決定
        agent.current_action = agent.decide_action(agent.current_pos, q_table, EPSILON)
        # print(q_table[agent.current_pos[0], agent.current_pos[1]])
        # print(agent.current_action)
        # 最大ステップ数に達するか、ゴールに到達するまでループ
        istep = 0
        while True:
            # 現時刻の行動から次時刻の位置を算出
            agent.move_agent(row=ROW, col=COL)
            # print("next_pos:{}".format(agent.next_pos))
            # 現時刻の行動と状態(現時刻の位置)から報酬を算出
            reward = agent.cal_reward(maze, row=ROW, col=COL)
            # print(reward)
            # show_map(maze, agent)
            # 次時刻の状態から次時刻の行動を決定
            agent.next_action = agent.decide_action(agent.next_pos, q_table, EPSILON)
            # 現時刻の状態で現時刻の行動を取った時のQ値を更新
            now_q = q_table[agent.current_pos[0]][agent.current_pos[1]][agent.current_action]
            # print("now_q:{}".format(now_q))
            now_x, now_y = agent.current_pos
            now_action = agent.current_action
            q_table[now_x, now_y, now_action] = update_q(q_table, agent, reward, ALPHA, GAMMA)
            # print("{} <- {} + {} * ({} + ({} * {}) - {}) = {}".format(now_q, now_q, ALPHA, reward, GAMMA, q_table[agent.next_pos[0], agent.next_pos[1]].max(), now_q, q_table[now_x, now_y, now_action]))
            # 状態を遷移させる
            agent.move_state(maze)
            istep += 1
            if (agent.current_pos == GOAL_POS) | (istep == NUM_STEP):
                break

    # 学習済みのQ値を外部ファイルに書き出し
    np.save(DIR+'/Q_table.npy', q_table)

    # 結果を描画するためにグリーディー(決定論的)に行動させる
    agent = Agent(START_POS)
    agent_log = [] # エージェントの軌跡を格納するリスト
    istep = 0
    while True:
        agent.current_action = np.argmax(q_table[agent.current_pos[0], agent.current_pos[1]])
        agent.move_agent(ROW, COL)
        agent_log.append(agent.current_pos)
        agent.move_state(maze)
        istep += 1
        if (agent.current_pos == GOAL_POS) | (istep == NUM_STEP):
            agent_log.append(agent.current_pos)
            break
    show_map(maze)
    plot_result(ROW, COL, np.array(agent_log))
              