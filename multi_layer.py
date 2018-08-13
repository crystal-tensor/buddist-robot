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
alphabase = './data/20182.csv'   # fof基金20170731-1031.csv   2018一季度基金数据2.csv
alphadata = './data/alpha_data.csv'
alphabasef = open(alphabase, 'r', encoding=u'utf-8', errors='ignore')
alphadataf = open(alphadata, 'r', encoding=u'utf-8', errors='ignore')
alphabasedf = pd.read_csv(alphabasef)
alphadatadf = pd.read_csv(alphadataf)
alphabasedf.dropna(inplace=True)
alphadatadf.dropna(inplace=True)
#conn = pymysql.Connect(host='192.168.101.10',port=3306,user='root',passwd='123456',db='stockcn',charset='utf8')
conn = pymysql.Connect(host='127.0.0.1',port=3306,user='root',passwd='',db='stockcn',charset='utf8')
cursor = conn.cursor()
sql = "select open_price from day_price where symbol=600000 ORDER BY day_price.price_date desc  limit 356"
df = pd.read_sql(sql=sql, con=conn)
stock = preprocessing.minmax_scale(df,feature_range=(-1,1))
#     stock = preprocessing.minmax_scale(stockdata,feature_range=(-1,1))
#     print(stock)

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

alpha = preprocessing.minmax_scale(alphadatadf.iloc[:, 7:17].values,feature_range=(-1,1))   #alpha的因变量
reward = preprocessing.minmax_scale(alphadatadf.iloc[:, 5:6].values,feature_range=(-1,1))   #收益率
moveloss = preprocessing.minmax_scale(alphadatadf.iloc[:, 27:28].values,feature_range=(-1,1))  #移动止损
acc_or_rej = preprocessing.minmax_scale(alphadatadf.iloc[:, 28:29].values,feature_range=(-1,1))  #客观的接受或者拒绝

x_data = preprocessing.minmax_scale(alphabasedf.iloc[:, 4:36].values, feature_range=(-1,1))
y_data = preprocessing.minmax_scale(alphabasedf.iloc[:, 3:4].values, feature_range=(-1,1))

#glable_PARAMS=1.68**k/math.exp(k) #定义全局变量·
#x_data, y_data = Variable(x_data), Variable(y_data)
# plt.scatter(x_data,y_data)
# plt.show()
#print(y_data)
xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing
emotion = addLayer(ys, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 情绪取决于移动止损的距离 值在-1到1之间
gb = addLayer(emotion/reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)  # 好和坏的初级判断或者叫初级推理，取决于收益和风险
ar = addLayer(reward/emotion, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 取舍逻辑，推理是否接受， 值在-1到1之间 ，值越接近0代表接受的程度越大
#firstmind = addLayer(ar/reward, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)   # 初心

#环境评估
firstmind = outputLayer(ar/reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 初心
rein_conscions = outputLayer(emotion/reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 结果和初心的距离
#acction = outputLayer(reward/moveloss, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 行为强化
output_logic = outputLayer(firstmind/ar, input_size, 1,  n_layer=1, activation_function=tf.nn.tanh)   # 结果和推理的距离
result_alpha = outputLayer(ar, input_size, 1,  n_layer=1, activation_function=tf.nn.tanh)   # 分类和推理的距离  任何的分类都是一种方法
position = addLayer(output_logic/result_alpha, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)

loss_position = tf.reduce_mean(tf.reduce_sum(tf.square((ys - position)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
train_position = tf.train.GradientDescentOptimizer(lr).minimize(loss_position)

#自我评估：
# reinforceconscious = addLayer(firstmind / (reward - emotion), input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
# unconscious = addLayer(firstmind * reinforceconscious, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
# wisdow=addLayer(unconscious/reinforceconscious, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
new_cloass = addLayer(stock / firstmind, 1, 1, n_layer=1, activation_function=tf.nn.tanh)
newconscuous = addLayer(firstmind * moveloss * new_cloass, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
reinforceconscious = addLayer(firstmind / (reward - emotion), input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
unconscious = addLayer(firstmind * reinforceconscious * newconscuous, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
wisdow = addLayer(unconscious / reinforceconscious, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)

losswisdow = tf.reduce_mean(tf.reduce_sum(tf.square((emotion - wisdow)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
trainwisdow = tf.train.GradientDescentOptimizer(lr).minimize(losswisdow)

# xsaplha = tf.placeholder(tf.float32, [None, input_size32])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
# ysaplha = tf.placeholder(tf.float32, [None, 1])
# l1 = addLayer(xsaplha, input_size32, 1, n_layer=1, activation_function=tf.nn.tanh) # relu是激励函数的一种
# l2 = addLayer(l1, 1, 1,n_layer=0, activation_function=None)
# lossaplha = tf.reduce_mean(tf.reduce_sum(tf.square((ysaplha - l2)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
# trainaplha = tf.train.GradientDescentOptimizer(lr).minimize(lossaplha)  # 选择梯度下降法
#d1 = tf.layers.dropout(l1, rate=0.1, training=tf_is_training)

#d2 = tf.layers.dropout(l2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs

 # 选择梯度下降法


loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - emotion)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法
# d_out = tf.layers.dense(d2, 1)
# d_loss = tf.losses.mean_squared_error(ys, d_out)
# d_train = tf.train.AdamOptimizer(lr).minimize(d_loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)



for k in range(1,3):
     # stockcode = "select symbol from symbol where id="
     # stockcode += str(k)
     # cursor.execute(stockcode)
     # id = cursor.fetchone()  high_price,low_price,close_price,volume,amount
     sql = "select open_price from day_price,symbol where day_price.symbol=symbol.symbol and symbol.id="
     sql += str(k)
     sql += " ORDER BY day_price.price_date desc  limit 356"
     # cursor.execute(sql)
     # stockdata = cursor.fetchall()
     # data = cursor.fetchone()
     df = pd.read_sql(sql=sql, con=conn)
     stock = preprocessing.minmax_scale(df,feature_range=(-1,1))
     # new_cloass = addLayer(stock / firstmind, 1, 1, n_layer=1, activation_function=tf.nn.tanh)
     # newconscuous = addLayer(firstmind * moveloss * new_cloass, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
     # reinforceconscious = addLayer(firstmind / (reward - emotion), input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
     # unconscious = addLayer(firstmind * reinforceconscious * newconscuous, input_size, 1, n_layer=1,activation_function=tf.nn.tanh)
     # wisdow = addLayer(unconscious / reinforceconscious, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)
     #print(stock)
     for i in range(3200):
          #loss_alpha, train_alpha = sess.run([lossaplha, trainaplha], feed_dict={xsaplha: x_data, ysaplha: y_data})
          loss_emotion, trainr = sess.run([loss, train], feed_dict={xs: reward, ys: moveloss})
          loss_wisdom, traina_wisdom = sess.run([losswisdow, trainwisdow], feed_dict={xs: moveloss, ys: stock})
          loss_location, traina_location = sess.run([loss_position, train_position], feed_dict={xs: acc_or_rej, ys: stock})
          if i % 2000 == 0:
             #print("loss_aplha", sess.run(lossaplha, feed_dict={xsaplha: x_data, ysaplha: y_data}))
             print("loss_emotion",sess.run(loss, feed_dict={xs: reward, ys: moveloss}))
             print("loss_wisdom", sess.run(losswisdow, feed_dict={xs: moveloss, ys: stock}))
             print("loss_location", sess.run(loss_position, feed_dict={xs: acc_or_rej, ys: stock}))




with tf.name_scope('loss'):
    loss_emotion = tf.reduce_mean(tf.reduce_sum(tf.square(ys - emotion),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    loss_wisdom = tf.reduce_mean(tf.reduce_sum(tf.square(emotion - wisdow), reduction_indices=[1]))
    tf.summary.scalar('loss_wisdom', loss_wisdom)
    loss_location = tf.reduce_mean(tf.reduce_sum(tf.square((ys - position)), reduction_indices=[1]))
    tf.summary.scalar('loss_location', loss_location)

with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss_emotion)
        train_step2 = tf.train.GradientDescentOptimizer(lr).minimize(loss_wisdom)
        train_step3 = tf.train.GradientDescentOptimizer(lr).minimize(loss_location)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train_step, feed_dict={xs: reward, ys: moveloss})
    sess.run(train_step2, feed_dict={xs: moveloss, ys: stock})
    sess.run(train_step3, feed_dict={xs: acc_or_rej, ys: stock})
    #sess.run(train_step2, feed_dict={xsaplha: x_data, ysaplha: y_data})
    if i % 50 == 0:
        result = sess.run(merged,feed_dict={xs: reward, ys: moveloss})
        writer.add_summary(result, i)
        result2 = sess.run(merged, feed_dict={xs: moveloss, ys: stock})
        writer.add_summary(result2, i)
        result3 = sess.run(merged, feed_dict={xs: acc_or_rej, ys: stock})
        writer.add_summary(result3, i)
        # result2 = sess.run(merged, feed_dict={xsaplha: x_data, ysaplha: y_data})
        # writer.add_summary(result2, i)

# if int((tf.__version__).split('.')[1]) < 12 and int(
#             (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#             writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#             writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs