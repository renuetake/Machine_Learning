import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import bp


class test_bp(unittest.TestCase):
    def test_read_data(self):
        """
        test of read_data method
        """
        # numpy配列を返しているかテスト
        expected = type(np.array([1, 2, 3]))
        actual = type(bp.read_data())
        self.assertEqual(expected, actual)

        # data.datがなかった場合はOSErrorを投げているかテスト
        with self.assertRaises(OSError):
            bp.read_data(path='data.txt')

    def test_init_net(self):
        """
        test method of init_net
        """
        # 引数がint以外だったらValueErrorを投げる
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(error_input, error_input, error_input)
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(normal_input, error_input, error_input)
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(error_input, normal_input, error_input)
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(error_input, error_input, normal_input)
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(normal_input, normal_input, error_input)
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(error_input, normal_input, normal_input)
        with self.assertRaises(ValueError):
            error_input = 'test_input'
            normal_input = 10
            bp.init_net(normal_input, error_input, normal_input)

        # 入力層・隠れ層・出力層それぞれのユニットを入力してnumpy配列を返しているかテスト
        num_input = 2
        num_hidden = 3
        num_output = 1
        weight1 = np.arange(6).reshape((num_input, num_hidden))
        weight2 = np.arange(3).reshape((num_hidden, num_output))
        weight = [weight1, weight2]

        expected = type(weight)
        actual = type(bp.init_net(num_input, num_hidden, num_output))
        self.assertEqual(expected, actual)

        # 返り値が引数通りの形になっているかテスト
        num_input = 2
        num_hidden = 3
        num_output = 1
        expected1 = (num_input, num_hidden)
        return_list = bp.init_net(num_input, num_hidden, num_output)
        actual1 = return_list[0].shape
        # 重み1(入力層~隠れ層)のテスト
        self.assertEqual(expected1, actual1)
        # 重み2(隠れ層~出力層)のテスト
        expected2 = (num_hidden, num_output)
        actual2 = return_list[1].shape
        self.assertEqual(expected2, actual2)

        # 重みが(-0.5~0.5)の範囲で初期化されているかテスト
        num_input = 2
        num_hidden = 3
        num_output = 1
        np.random.seed(seed=10)
        return_list = bp.init_net(num_input, num_hidden, num_output, seed=10)
        # 重み1(入力層~隠れ層)のテスト
        expected1 = np.random.rand(num_input, num_hidden)
        expected1 = expected1.all() - 0.5
        actual1 = return_list[0]
        self.assertEqual(expected1.all(), actual1.all())
        # 重み2(隠れ層~出力層)のテスト
        expected2 = np.random.rand(num_hidden, num_output)
        expected2 = expected2.all() - 0.5
        actual2 = return_list[1]
        self.assertEqual(expected2.all(), actual2.all())

    def test_feedforward(self):
        """
        test method of feedforward
        """
        # 引数のweightがリスト、dataがnp.array以外のisampleがint以外のデータ型が入力されたらValueErrorを投げているかテスト
        with self.assertRaises(ValueError):
            error_input_int = 10
            error_input_str = 'test'
            bp.feedforward(error_input_str, error_input_int, error_input_int)
        
        # 出力の型が正しいかテスト
        # num_input=2, num_hidden=3, num_output=1の場合の重みのリスト
        weight = [[[1,1,1],[1,1,1]],[[1],[1],[1]]]
        # 入力=2, 出力=1のdata
        data = np.array([[0,0,1],[0,1,0]])
        isample = 0
        expected = type([[1,2],[1,2]])
        actual = type(bp.feedforward(weight, data, isample))
        self.assertEqual(expected, actual)

        # 出力が正しいかどうかテスト
        # num_input=2, num_hidden=3, num_output=1の場合の重みのリスト
        weight = [[[1,1,1],[1,1,1]],[[1],[1],[1]]]
        # 入力=2, 出力=1のdata
        data = np.array([[0,0,1],[0,1,0]])
        isample = 0
        h_sig = (1.0 / (1.0 + np.exp(0 * -0.8)))
        o_sig = (1.0 / (1.0 + np.exp((h_sig*3) * -0.8)))
        expected = [[0,0,1.0], [h_sig,h_sig,h_sig,1.0], [o_sig]]
        actual = bp.feedforward(weight, data, isample)
        self.assertEqual(expected, actual)

    def test_backward(self):
        """
        test method of backward
        """
        # 引数のweightがリスト、dataがnp.array以外のisampleがint以外のデータ型が入力されたらValueErrorを投げているかテスト
        with self.assertRaises(ValueError):
            error_input_int = 10
            error_input_str = 'test'
            bp.backward(error_input_str, error_input_int, error_input_int, error_input_str)
        
        # 出力の型が正しいかテスト
        # num_input=2, num_hidden=3, num_output=1の場合の重みのリスト
        weight = [[[1,1,1],[1,1,1]],[[1],[1],[1]]]
        # 入力=2, 出力=1のdata
        data = np.array([[0,0,1],[0,1,0]])
        isample = 0
        out = [[0, 0, 1.0], [0.5, 0.5, 0.5, 1.0], [0.7685247834990178]]
        expected = type([[1,2],[1,2]])
        actual = type(bp.backward(weight, data, isample, out))
        self.assertEqual(expected, actual)

        # 出力が正しいかどうかテスト
        # num_input=2, num_hidden=3, num_output=1の場合の重みのリスト
        weight = [[[1,1,1],[1,1,1]],[[1],[1],[1]]]
        # 入力=2, 出力=1のdata
        data = np.array([[0,0,1],[0,1,0]])
        isample = 0
        out = [[0, 0, 1.0], [0.5, 0.5, 0.5, 1.0], [0.7685247834990178]]
        o_back = 0.8 * (1.0 - 0.7685247834990178) * (1.0 - 0.7685247834990178) * 0.7685247834990178
        h_back = 0.8 * (o_back) * (1.0 - 0.5) * 0.5
        expected = [[h_back,h_back,h_back], [o_back]]
        actual = bp.backward(weight, data, isample, out)
        self.assertEqual(expected, actual)
        
    def test_modify_weights(self):
        """
        test method of modify_weights
        """
        # 入力された型がリスト以外のときValueErrorを投げているかテスト
        with self.assertRaises(ValueError):
            NG = 10
            OK = []
            bp.modify_weights(NG, OK, NG)




