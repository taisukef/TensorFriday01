# coding: UTF-8
# 2層のニューラルネットワーク

import tensorflow as tf
import numpy as np

# ファイルからndarrayを取得
def open_with_numpy_loadtxt(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

x = tf.placeholder("float",[None, 4]) # 入力層
y_ = tf.placeholder("float",[None, 6]) # 出力層

num = 20

# Tensorのsize mean: 平均 stdev: 標準偏差
w_h = tf.Variable(tf.random_normal([4, num], mean=0.0, stddev=0.05))
b_h = tf.Variable(tf.zeros([num]))
w_o = tf.Variable(tf.random_normal([num, 6], mean=0.0, stddev=0.05))
b_o = tf.Variable(tf.zeros([6]))

# model
def model(x, w_h, b_h, w_o, b_o):
    # matmulは行列の掛け算をしてくれる
    # sigmoidは活性化関数
    h = tf.sigmoid(tf.matmul(x, w_h) + b_h)
    # Reluで活性化したい場合
    #h = tf.nn.relu(tf.matmul(x, w_h) + b_h)

    # softmax関数は与えたデータから確率っぽいものを求めてくれます
    # softmax関数も活性化関数です
    return tf.nn.softmax(tf.matmul(h, w_o) + b_o)

testData = tf.convert_to_tensor(open_with_numpy_loadtxt("testRGBYB.csv"))
testLabel = tf.convert_to_tensor(open_with_numpy_loadtxt("testLabelsRGBYB.csv"))

# 重さやバイアスを保存してくれる
saver = tf.train.Saver()

with tf.Session() as sess:
    # 初期化
    sess.run(tf.global_variables_initializer())

    # セーブした変数を復元
    saver.restore(sess, "./model.ckpt")
    print("Model restored.")

    testData = tf.cast(testData.eval(), tf.float32)
    y_hypo = model(testData.eval(), w_h, b_h, w_o, b_o)
    y_hypo = tf.cast(y_hypo.eval(), tf.float64)

    # 分類結果の判定
    correct_prediction = tf.equal(tf.argmax(y_hypo,1), tf.argmax(y_,1))
    # 精度の計算
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
    print("学習結果")
    print("test dataでの精度")
    print(accuracy.eval({x: testData.eval(), y_: testLabel.eval()}) * 100)

    result = tf.nn.softmax(y_hypo.eval())
    print("result:")
    # 一番大きい値のindexがその判定した色の番号
    print(result[0:10].eval())

