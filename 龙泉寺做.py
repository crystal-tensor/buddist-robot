
import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
#import math

# hidden layer
rnn_unit = 128
# feature
input_size = 53
output_size = 1
lr = 0.0006
csv_file = 'stock20170331-0721.csv'
csv_file2 = 'EURUSD60.csv'
f = open(csv_file, 'r', encoding=u'utf-8', errors='ignore')
fmoveloss = open(csv_file2, 'r', encoding=u'utf-8', errors='ignore')
df = pd.read_csv(f)
dfmoveloss = pd.read_csv(fmoveloss)
df.dropna(inplace=True)
dfmoveloss.dropna(inplace=True)

def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

alpha = preprocessing.minmax_scale(df.iloc[:, 3:56].values,feature_range=(-1,1))
income = preprocessing.minmax_scale(df.iloc[:, 56:57].values,feature_range=(-1,1))
moveloss = preprocessing.minmax_scale(dfmoveloss.iloc[:, 13:14].values,feature_range=(-1,1))
undulate_rate = 30%


#glable_PARAMS=1.68**k/math.exp(k) #定义全局变量
#x_data, y_data = Variable(x_data), Variable(y_data)
# plt.scatter(x_data,y_data)
# plt.show()

xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing
gb= addLayer(income,undulate_rate, input_size, 1, activity_function=tf.nn.tanh) # 好和坏的初级判断取决于收益和风险

emotion = addLayer(moveloss, input_size, 1, activity_function=tf.nn.tanh) # 情绪取决于移动止损的距离 值在-1到1之间

def acc_or_rej(emotion,imcome,undulate_rate)  #接受还是拒绝
    if emotion==0 acc_or_rej=gb
        else
    if emotion >0 then  acc_or_rej=1
    if emotion < 0 then acc_or_rej=0
    return if acc_or_rej

#judgement:

firstmind = addLayer(acc_or_rej/income的涨跌, input_size, 1, activity_function=tf.nn.tanh) # relu是激励函数的一种
rein_conscions = addLayer(emotion/income*100%, input_size, 1, activity_function=tf.nn.tanh) # relu是激励函数的一种
acction = addLayer(income/moveloss, input_size, 1, activity_function=tf.nn.tanh) # relu是激励函数的一种
output_logic =
结果和分类的距离=gb/income的好坏率

#self：
reinforceconscious=firstmind／income-emotion
unconscious=love+gene_defect   #无意识来自爱和基因缺陷
newconscuous=moveloss>1   #新的思想来自新的方法，也就是新的分类，另外还取决于压力的大小和初心的大小
subconscious=lossrate=var/MathAbs(balanceall)*100   #潜意识就是潜在风险的累积值

#逻辑输出：包括反思和推理

#任务：比如提高alpha的收益率

#新思想的产生 或者新方法的推演

#内部对冲和外部对冲 新思想大部分来自外部的刺激



d1 = tf.layers.dropout(l1, rate=0.5, training=tf_is_training)
l2 = addLayer(l1, 1, 1, activity_function=None)
d2 = tf.layers.dropout(l2, rate=0.5, training=tf_is_training)   # drop out 50% of inputs

loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)  # 选择梯度下降法

d_out = tf.layers.dense(d2, 1)
d_loss = tf.losses.mean_squared_error(ys, d_out)
d_train = tf.train.AdamOptimizer(lr).minimize(d_loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

for i in range(100):
    #sess.run(train,feed_dict={xs: x_data, ys: y_data})
    loss_overfiting,trainr=sess.run([loss,train], feed_dict={xs: x_data, ys: y_data})
    #sess.run(d_train, feed_dict={xs: x_data, ys: y_data,tf_is_training: True})
    loss_dropout, d_trainr = sess.run([d_loss, d_train], feed_dict={xs: x_data, ys: y_data,tf_is_training: True})
    #acc = np.average(np.abs(ys -l2) / ys)  # 偏差
    #base_path = saver.save(sess, "module/alpha520-3.model")
    if i % 20 == 0:
        print("loss_overfiting",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        print("loss_dropout", sess.run(d_loss, feed_dict={xs: x_data, ys: y_data,tf_is_training: True}))
#         plt.cla()
#         plt.text(0.1, 0, 'loss_overfiting=%.04f' % loss_overfiting, fontdict={'size': 20, 'color': 'red'})
#         plt.text(0.1, 1, 'loss_dropout=%.04f' % loss_dropout, fontdict={'size': 20, 'color': 'green'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)


# $ tensorboard --logdir=logs