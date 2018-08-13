###宏观数据对应债券
#对位置的把握才表明是对行情的把握。就是在宏观数据变坏之前，之后，之中能退出

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import keras
import pymysql


# hidden layer
rnn_unit = 128
# feature
input_size = 1
output_size = 1
input_size32 = 32
lr = 0.0006

#conn = pymysql.Connect(host='192.168.101.10',port=3306,user='root',passwd='123456',db='stockcn',charset='utf8')
conn = pymysql.Connect(host='127.0.0.1',port=3306,user='root',passwd='',db='stockcn',charset='utf8')
cursor = conn.cursor()

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
def outputLayer(inputs, in_size, out_size, n_layer, activation_function=None):
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

xsfinance = tf.placeholder(tf.float32, [None, 6])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
#xsday = tf.placeholder(tf.float32, [None, 6])
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)   # to control dropout when training and testing
#emotion = addLayer(xsday, 6, 1, n_layer=1, activation_function=tf.nn.tanh)  # 情绪取决于移动止损的距离 值在-1到1之间
#ed1 = addLayer(emotion, 1, 1, n_layer=1, activation_function=tf.nn.tanh)

def selfassess():     # painting from the famous artist (real target)
    wisdow = addLayer(xsfinance, 6, 1, n_layer=1, activation_function=tf.nn.tanh)
    ew1 = addLayer(wisdow, 1, 1, n_layer=1, activation_function=tf.nn.tanh)
    d2 = tf.layers.dropout(ew1, rate=0.1, training=tf_is_training)
    return d2
correct_prediction = tf.equal(tf.argmax(ys,1),tf.argmax(selfassess(),1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#firstmind = outputLayer(ar/stockday, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 初心
#环境评估
def environment():     # painting from the famous artist (real target)
    rein_conscions = outputLayer(emotion / reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 结果和初心的距离
    # 行为强化 action = outputLayer(reward / moveloss, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 行为强化
    output_logic = outputLayer(firstmind / ar, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 结果和推理的距离
    result_alpha = outputLayer(ar, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 分类和推理的距离  任何的分类都是一种方法
    position = addLayer(output_logic / result_alpha, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
    return position

# loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - ed1)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
# train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法

losswisdow = tf.reduce_mean(tf.reduce_sum(tf.square(ys - selfassess()), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
trainwisdow = tf.train.GradientDescentOptimizer(lr).minimize(losswisdow)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)



for k in range(1,120):
     sqlday = "select open_price,high_price,low_price,close_price,volume,amount from day_price,symbol where day_price.symbol=symbol.symbol and symbol.id="
     sqlday += str(k)
     sqlday += " order by price_date desc limit 252"
     # sqlmacro = "SELECT qianzhi,gongbuzhi from macro_economy where country="
     # sqlmacro += "'china'"
     # sqlmacro += " and shijian like "
     # sqlmacro += "'2017%'"
     # sqlmacro += " GROUP BY shijian"
     sql_y = "select close_price from bond"
     sql_y += " ORDER BY date desc limit 252"
     dfsqlday= pd.read_sql(sql=sqlday, con=conn)
     #print(dfsqlday)
     df_y = pd.read_sql(sql=sql_y, con=conn)
     x_data = preprocessing.minmax_scale(dfsqlday, feature_range=(-1,1))
     y_data = preprocessing.minmax_scale(df_y, feature_range=(-1, 1))
     #print(x_data)
     #selfassess()
     for i in range(3200):
          #loss_alpha, train_alpha = sess.run([lossaplha, trainaplha], feed_dict={xsaplha: x_data, ysaplha: y_data})
          loss_wisdom, traina_wisdom = sess.run([losswisdow, trainwisdow], feed_dict={xsfinance: x_data, ys: y_data, tf_is_training: True})
          #loss_emotion, trainr = sess.run([loss, train], feed_dict={xsday: stockday, ys: stock_y})
          acc = sess.run(accuracy, feed_dict={xsfinance: x_data, ys: y_data, tf_is_training: True})
          if i % 3000 == 0:
             #print("loss_aplha", sess.run(loss_emotion, feed_dict={xsday: stockday, ys: stock_y}))
            # print("loss_emotion",sess.run(loss, feed_dict={xsday: stockday, ys: stock_y}))
             print("loss_wisdom", sess.run(losswisdow, feed_dict={xsfinance: x_data, ys: y_data, tf_is_training: True}))
             #print("accuracy", sess.run(accuracy, feed_dict={xs: moveloss, ys: stockday}))
             print("accuracy", sess.run(accuracy, feed_dict={xsfinance: x_data, ys: y_data, tf_is_training: True}))



with tf.name_scope('loss'):
    loss_wisdom = tf.reduce_mean(tf.reduce_sum(tf.square(ys - selfassess()), reduction_indices=[1]))
    tf.summary.scalar('loss_wisdom', loss_wisdom)


with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss_wisdom)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(30000):
    sess.run(train_step, feed_dict={xsfinance: x_data, ys: y_data,tf_is_training: True})
    #sess.run(train_step2, feed_dict={xsaplha: x_data, ysaplha: y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xsfinance: x_data, ys: y_data,tf_is_training: True})
        writer.add_summary(result, i)


# if int((tf.__version__).split('.')[1]) < 12 and int(
#             (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#             writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#             writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs