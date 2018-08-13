

from __future__ import print_function
import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
# hidden layer
rnn_unit = 128
# feature
input_size = 1
output_size = 1
lr = 0.0006
csv_file = './data/alpha_data.csv'
f = open(csv_file, 'r', encoding=u'utf-8', errors='ignore')
df = pd.read_csv(f)
df.dropna(inplace=True)

def addLayer(inputs, in_size, out_size, n_layer, activation_function=None):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
        tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
def addLayer2(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

alpha = preprocessing.minmax_scale(df.iloc[:, 7:17].values,feature_range=(-1,1))   #alpha的因变量
income = preprocessing.minmax_scale(df.iloc[:, 5:6].values,feature_range=(-1,1))   #收益率
moveloss = preprocessing.minmax_scale(df.iloc[:, 27:28].values,feature_range=(-1,1))  #移动止损
acc_or_rej = preprocessing.minmax_scale(df.iloc[:, 28:29].values,feature_range=(-1,1))  #客观的接受或者拒绝
undulate_rate = 0.3   #波动率

xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing
emotion = addLayer(ys, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 情绪取决于移动止损的距离 值在-1到1之间
gb = addLayer(emotion/income, input_size, 1,n_layer=1,activation_function=tf.nn.tanh)  # 好和坏的初级判断或者叫初级推理，取决于收益和风险
ar = addLayer(income/emotion, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)   # 取舍逻辑，推理是否接受， 值在-1到1之间 ，值越接近0代表接受的程度越大
firstmind = addLayer(ar/income, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)   # 初心
#alpha_neural = addLayer2(alpha, input_size,  1, activity_function=tf.nn.tanh)

#self：
reinforceconscious=addLayer(firstmind/(income-emotion), input_size, 1 ,n_layer=1,activation_function=tf.nn.tanh)
# unconscious=love+gene_defect   #无意识来自爱和基因缺陷
# newconscuous=moveloss>1   #新的思想来自新的方法，也就是新的分类，另外还取决于压力的大小和初心的大小
# subconscious=lossrate=var/MathAbs(balanceall)*100   #潜意识就是潜在风险的累积值

#d1 = tf.layers.dropout(emotion, rate=0.1, training=tf_is_training)
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys- emotion)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
#tf.contrib.layers.l2_regularizer(lambda)(w)  #正则化
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
#writer = tf.summary.FileWriter("logs/", sess.graph)

for i in range(5000):
    lossm, trainm = sess.run([loss, train], feed_dict={xs:income , ys:moveloss })
    #loss_dropout_buy, buy_trainer = sess.run([buy_loss, buy_train], feed_dict={xs: buy_data_movelosss, ys: y_data,tf_is_training: True})
    #loss_overfiting_sell, sell_trainer = sess.run([loss_sell, sell_train], feed_dict={xs: sell_data_movelosss, ys: y_data, tf_is_training: True})
    #base_path = saver.save(sess, "module/emotion.model")
    if i % 1000 == 0:
        #print("loss_overfiting",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        print("lossm", sess.run(loss, feed_dict={xs: income, ys: moveloss,tf_is_training: True}))
        #print("loss_overfiting_sell",sess.run(loss_sell, feed_dict={xs: buy_data_movelosss, ys: y_data, tf_is_training: True}))
        #result = sess.run(loss_buy, feed_dict={xs: x_data, ys: y_data})
        #writer.add_summary(loss,i)

#         plt.cla()
#         plt.text(0.01, 0, 'loss_overfiting=%.04f' % loss_overfiting_buy, fontdict={'size': 20, 'color': 'red'})
#         #plt.text(0.01, 1, 'loss_dropout=%.04f' % loss_dropout, fontdict={'size': 20, 'color': 'green'})
#         plt.pause(0.001)
#
# plt.ioff()
# plt.show()

with tf.name_scope('lossm'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - emotion), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('trainm'):
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(5000):
    sess.run(train_step, feed_dict={xs: income, ys: moveloss})
    if i % 50 == 0:
        result = sess.run(merged,
                          feed_dict={xs: income, ys: moveloss})
        writer.add_summary(result, i)


# if int((tf.__version__).split('.')[1]) < 12 and int(
#                (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#                writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#                writer = tf.summary.FileWriter("logs/", sess.graph)

#tensorboard --logdir=logs





