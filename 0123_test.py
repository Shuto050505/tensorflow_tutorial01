
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series
import cv2
import sys


# In[5]:

CHAN = 1
PIX = 28
IMAGE_PIXELS = PIX ** 2
NUM_CLASSES = 10
FILT = Series({'N':3,'num1':32,'num2':64,'num3':1024})
FLAG = Series({'max_steps':15000,'learning_rate':1e-4,'train_dir':'data','batch_size':100,'test_size':2000})


# In[ ]:

# 重みを標準偏差0.1の正規分布で初期化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアスを標準偏差0.1の正規分布で初期化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み層の作成
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# プーリング層の作成
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# In[3]:

def inference(images_placeholder, keep_prob):
    """ 予測モデルを作成する関数

    引数: 
      images_placeholder: 画像のplaceholder
      keep_prob: dropout率のplaceholder

    返り値:
      y_conv: 各クラスの確率(のようなもの)
    """

    # 入力を変形
    x_image = tf.reshape(images_placeholder, [-1, PIX, PIX, CHAN])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([FILT.N, FILT.N, CHAN, FILT.num1])
        b_conv1 = bias_variable([FILT.num1])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        
    # 畳み込み層1aの作成
    with tf.name_scope('conv1a') as scope:
        W_conv1a = weight_variable([FILT.N, FILT.N, FILT.num1, FILT.num1])
        b_conv1a = bias_variable([FILT.num1])
        h_conv1a = tf.nn.relu(conv2d(h_conv1, W_conv1a) + b_conv1a)

    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1a)
    
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([FILT.N, FILT.N, FILT.num1, FILT.num2])
        b_conv2 = bias_variable([FILT.num2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # 畳み込み層2aの作成
    with tf.name_scope('conv2a') as scope:
        W_conv2a = weight_variable([FILT.N, FILT.N, FILT.num2, FILT.num2])
        b_conv2a = bias_variable([FILT.num2])
        h_conv2a = tf.nn.relu(conv2d(h_conv2, W_conv2a) + b_conv2a)
        
    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2a)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        num4 = int((PIX/4)*(PIX/4)*FILT.num2)
        W_fc1 = weight_variable([num4, FILT.num3])
        b_fc1 = bias_variable([FILT.num3])
        h_pool2_flat = tf.reshape(h_pool2, [-1, num4])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([FILT.num3, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv


# In[4]:

if __name__ == '__main__':
    test_image = []
    for i in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[i])
        img = cv2.resize(img, (PIX, PIX))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        test_image.append(img.flatten().astype(np.float32)/255.0)
    test_image = 1 - np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder(tf.float32)

    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "kept\model.ckpt")

    for i in range(len(test_image)):
        pred = np.argmax(logits.eval(feed_dict={ 
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0])
        print('Answer: ',pred)


# In[ ]:



