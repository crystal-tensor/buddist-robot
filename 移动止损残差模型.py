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
csv_file = './data/EURUSD60.csv'
f = open(csv_file, 'r', encoding=u'utf-8', errors='ignore')
df = pd.read_csv(f)
df.dropna(inplace=True)

# n=97198
# moveloss=0.00120
#y_data = preprocessing.minmax_scale(df.iloc[:, 10:11].values,feature_range=(-1,1))
y_data = preprocessing.minmax_scale(df.iloc[:, 5:6].values,feature_range=(-1,1))
#sell_data_movelosss = df.iloc[:, 13:14].values
#x_data = preprocessing.minmax_scale(df.iloc[:, 13:14].values,feature_range=(-1,1))
x_data = preprocessing.minmax_scale(df.iloc[:, 18:19].values,feature_range=(-1,1))

#print(sell_data_movelosss)
xs = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing
# close = df.iloc[:, 13:14]
# buy_movelosss=df.iloc[:, 10:11]
# print(buy_movelosss[n:])
# X=open, close,high,low,speed_5,speed30,speed1h,speed4h
# Y=moveloss(open+setfirstloss=50)

def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

l1 = addLayer(xs, input_size, 1, activity_function=tf.nn.tanh) # relu是激励函数的一种
d1 = tf.layers.dropout(l1, rate=0.1, training=tf_is_training)
#l2 = addLayer(l1, 1, 1, activity_function=None)
#d2 = tf.layers.dropout(l2, rate=0.1, training=tf_is_training)   # drop out 50% of inputs

# l3 = addLayer(xs, input_size, 1, activity_function=tf.nn.tanh) # relu是激励函数的一种
# d3 = tf.layers.dropout(l3, rate=0.1, training=tf_is_training)
# l4 = addLayer(l3, 1, 1, activity_function=None)
# d4 = tf.layers.dropout(l4, rate=0.1, training=tf_is_training)   # drop out 50% of inputs
# ACTIONS = ['up ','down']
# # observation_, reward, done = env.step(action)
# m,reward=1
# if  close[n+m:]==(buy[n+m:]-moveloss):
#
#    Else if high0-high1<0 moveloss=setfirstloss
#              Else movelosss= speed30/*setfirstloss+action
#                 reward = reward+1
#                 周期up=周期+1
#            if high0-high1<0 moveloss=setfirstloss
#               Else movelosss= speed30/*setfirstloss+action
#                 reward = reward-1
#                 周期totle=周期totle+1
#自我强化学习
# def build_q_table(n_states, actions):
#     table = pd.DataFrame(
#         np.zeros((n_states, len(actions))),     # q_table initial values全0初始化
#         columns=actions,    # actions's name
#     )
#     # print(table)    # show table
#     return table
#
# N_STATES = close[n]   # 第一次到close的bar的距离，比如可以从第一根k线开始
# ACTIONS = ['up', 'down']     # available actions
# EPSILON = 0.9   # greedy police就是说90%的时候选择最优的动作，10%选随机的动作
# ALPHA = 0.1     # learning rate学习率
# GAMMA = 0.9    # discount factor未来奖励的衰减度
# MAX_EPISODES = 13   # maximum episodes总共玩多少回合
# FRESH_TIME = 0.3    # fresh time for one move每走一步花多长时间
#
# def choose_action(state, q_table):
#     # This is how to choose an action
#     state_actions = q_table.iloc[state, :]
#     if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # act non-greedy or state-action have no value
#         action_name = np.random.choice(ACTIONS)
#     else:   # act greedy
#         action_name = state_actions.argmax()
#     return action_name
#
# def get_env_feedback(S, A):
#     # This is how agent will interact with the environment
#     if A == 'up':    # move up
#         if S = x_data[]
#            if(close-open>0):   # terminate  k线往后走一根
#             S_up = close[n]-((close[n]-close[n-1])*0.618+moveloss)  #S就是状态
#             R_up = 1  #R就是reward
#         else:
#             S_ = S + 1
#             R = 0
#     else:   # move down
#         R = 0
#         if S ==  close[n]+1:
#             if(close-open<0):
#             S_down = close[n]+((close[n]-close[n-1])*0.618+moveloss)  # reach the wall
#             R_down = -1
#         else:
#             S_ = S - 1
#     return S_, R
#
# def update_env(S, episode, step_counter):
#     # This is how environment be updated
#     env_list = close[n]  # our environment
#     if S == S_up-S_down>0: 如果涨的moveloss比跌的moveloss还要高，说明可以平仓多单了，反之，如果跌的movelss比涨的moveloss
#         interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
#         print('\r{}'.format(interaction), end='')
#         time.sleep(2)
#         print('\r ', end='')
#     else:
#         env_list[S] = close[n]
#         interaction = ''.join(env_list)
#         print('\r{}'.format(interaction), end='')
#         time.sleep(FRESH_TIME)


#外汇的核心比的是多空双方止损的距离，如果多方的止损比空方的止损距离大说明行情可能是在跌的，反之，如果空方的止损比多方的止损大说明行情可能是在涨的
#重点是如果y=buy的止损/sell的止损距离>x,x是某个阈值，那么行情可能要发生逆转。
#需要解决的问题是：这个y在loss里面的实际值是什么？
# 最小化今天的亏损
loss_buy = tf.reduce_mean(tf.reduce_sum(tf.square((ys- l1)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
#tf.contrib.layers.l2_regularizer(lambda)(w)  #正则化
train_buy = tf.train.GradientDescentOptimizer(lr).minimize(loss_buy)  # 选择梯度下降法
buy_out = tf.layers.dense(d1, 1)
buy_loss = tf.losses.mean_squared_error(ys, buy_out)
buy_train = tf.train.AdamOptimizer(lr).minimize(buy_loss)
# 最大化截止到昨天的收益
# loss_sell = tf.reduce_mean(tf.reduce_sum(tf.square((ys- l4)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
# train_sell = tf.train.GradientDescentOptimizer(lr).minimize(loss_sell)  # 选择梯度下降法
# sell_out = tf.layers.dense(d4, 1)
# sell_loss = tf.losses.mean_squared_error(ys, sell_out)
# sell_train = tf.train.AdamOptimizer(lr).minimize(sell_loss)
#train=optimizer.minmize(loss)
#交叉熵计算loss
# y_=tf.placeholder(tf.float32,[none,10])
# corss_entropy=-tf.reduce_sum(y_ * tf.log(y))
# train_step=tf.grain.gradientDescentOptimizer(0.01).minimize(cross_entropy)
# 计算准确率
# y=df.iloc[:, 8:9]
# correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction.tf.float32))

#train
 # tf.global_variables_initializer().run()
 #   for i in range(500):
 #         batch_xs,batch_ys=mnist.train.next_batch(50)
 #         train_step.run({batch_xs,y_:batch_ys})

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
#writer = tf.summary.FileWriter("logs/", sess.graph)

for i in range(100):
    loss, trainr = sess.run([loss_buy, train_buy], feed_dict={xs: x_data, ys: y_data})
    #loss_dropout_buy, buy_trainer = sess.run([buy_loss, buy_train], feed_dict={xs: buy_data_movelosss, ys: y_data,tf_is_training: True})
    #loss_overfiting_sell, sell_trainer = sess.run([loss_sell, sell_train], feed_dict={xs: sell_data_movelosss, ys: y_data, tf_is_training: True})
    #base_path = saver.save(sess, "module/forex_moveloss2.model")
    if i % 200 == 0:
        #print("loss_overfiting",sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        print("loss", sess.run(loss_buy, feed_dict={xs: x_data, ys: y_data,tf_is_training: True}))
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

with tf.name_scope('loss'):
    loss_buy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - l1),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss_buy)

with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss_buy)

sess = tf.Session()
merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)


# if int((tf.__version__).split('.')[1]) < 12 and int(
#                (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
#                writer = tf.train.SummaryWriter('logs/', sess.graph)
# else:  # tensorflow version >= 0.12
#                writer = tf.summary.FileWriter("logs/", sess.graph)

#tensorboard --logdir=logs


















