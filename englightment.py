import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math


# rnn_unit = 128
# input_size = 1
# output_size = 1
# lr = 0.0006
# csv_file = './data/EURUSD60.csv'
# f = open(csv_file, 'r', encoding=u'utf-8', errors='ignore')
# df = pd.read_csv(f)
# df.dropna(inplace=True)

#五毒的初始化，可改为十不善业或者比丘尼戒中的不善业
corrupt=t = np.random.uniform(-1, 0, size=BATCH_SIZE)[:, np.newaxis]
hate=np.random.uniform(-1, 0, size=BATCH_SIZE)[:, np.newaxis]
delusion=np.random.uniform(-1, 0, size=BATCH_SIZE)[:, np.newaxis]
Pride=np.random.uniform(-1, 0, size=BATCH_SIZE)[:, np.newaxis]
suspect=np.random.uniform(-1, 0, size=BATCH_SIZE)[:, np.newaxis]
#业的初始值
Karma=tf.Variable(tf.random_normal())
#五蕴的初始化
eye=np.random.uniform(0, 1, size=BATCH_SIZE)[:, np.newaxis]
hear=np.random.uniform(0, 1, size=BATCH_SIZE)[:, np.newaxis]
nose=np.random.uniform(0, 1, size=BATCH_SIZE)[:, np.newaxis]
tongue=np.random.uniform(0, 1, size=BATCH_SIZE)[:, np.newaxis]
body=np.random.uniform(0, 1, size=BATCH_SIZE)[:, np.newaxis]
#八识
aware=tf.Variable(tf.random_normal())
Manas=tf.Variable(tf.random_normal())


heart_mind_data = preprocessing.minmax_scale(df.iloc[:, 1:2].values,feature_range=(-1,1))
Karma_data = preprocessing.minmax_scale(df.iloc[:, 1:256].values,feature_range=(-1,1))

Karma = tf.placeholder(tf.float32, [None, input_size])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
heart_mind = tf.placeholder(tf.float32, [None, 1])
tf_is_training = tf.placeholder(tf.bool, None)  # to control dropout when training and testing

def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.random_normal([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans

Alaya=(W*(eye,hear,nose,tongue,body,aware),b) #阿赖耶识应该是个数组，分为善，恶和非善非恶三个数组，用softmax区分善和恶，随机数组为非善非恶

feed=F(W*Alaya,b)
#因果论的逻辑--非线性逻辑回归
#feed是因，acction是果，karma是业
Karma=F(W*(feed,acction),b) #“种子起现行，现行熏种子”
#因果论的自我进化其实用了深度强化学习
reward=F(acction)
#邪淫，吸毒等能给身体和意识带来愉快的是因为
reward=happy_low(acction)

#悟空就是觉醒，数学上用镜像函数来表示，佛为什么说每个人都有佛性，佛性其实是指的阿赖耶识的镜像函数，如来藏其实就是镜像中的阿赖耶识，
# 下面是简单的模型验证
mirror_Alaya==1，其实mirror_Alaya（佛性）一直存在
#所以觉醒，证悟空性，其实是指
anti_Alaya=mirror(Alaya)
#证悟前
eye=eye #看山是山
#证悟后
eye=mirror(eye) #看山还是山，是照见五蕴，但看到东西还是那些东西，差别就在镜子里的东西是不存入阿赖耶识的，一旦存入到阿赖耶识就意味着feed就产出了，
#  就是有了"因"

mirror_Alaya=mirror(Alaya)
def mirror(Alaya):
mirrorseye = mirror(eye)
mirrorshear = mirror(hear)
mirrorsnose = mirror(nose)
mirrorstongue = mirror(tongue)
mirrorsbody = mirror(body)
corrupt=hate=delusion=Pride=suspect=0=format(eye,hear,nose,tongue,body)

#空性当中其实包含了因果论
emptiness=mirror_Alaya #你给mirror函数好的返回来的就是好的，你给镜像函数不好的反回来就是不好的。所谓种瓜得瓜，种豆得豆。
#愿力是业的函数，
Vow_Power=F(Karma)

#证悟空性--成佛（3种方法）
#方法1：闻思修(本尊修法。历史上很多成就者就是修本尊获得成就。84位成就者，包括龙树菩萨、米拉日巴都是这样，通过修一个菩萨、本尊最后成就的。
# )。修行其实是自我强化学习业果的梯度下降，如果能收敛下去就能有成就，成就就是五毒减少，tathagatagarbha被发现，启动mirror函数，。
Karma = tf.reduce_mean(tf.reduce_sum(tf.square((real-Vow_Power)) #Vow_Power是愿力，real是真实
Practice = tf.train.GradientDescentOptimizer(lr).minimize(Karma) # 选择梯度下降法，收敛的最低点就是镜像所在。lr是学习率，
# 每个人天赋和努力程度是不一样的，所以lr=F(Talent，Effort)
tantric=tf.train.AdamOptimizer(lr).minimize(Karma) #密法我的理解是修行的方便法门，数学上就是对实修的优化算法
#修行多久能开悟呢？初始化-皈依
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
#开始修行（过程是：皈依-按仪轨修行-回向）
for i in range(times):
#loss_overfiting是衡量修行跑偏的程度
Karma_loss,trainr=sess.run([loss,train], feed_dict={xs: x_data, ys: y_data})
#loss_dropout是矫正修行跑偏的函数
heart-mind_loss, d_trainr = sess.run([d_loss, d_train], feed_dict={xs: x_data, ys: y_data,tf_is_training: True})
#acc = 准确率反应的是修行的效果，越接机于1代表修行的越完美 # 偏差
base_path = saver.save(sess, "mirror_all/mirror_Alaya.model")
if(Alaya =mirror(Alaya)) #末那识如果对五蕴不做转换，不把看到，听到的东西存到alaiyeshi里面，那么alaiyeshi就类似一个空数据库，
# 这样就不会产生因feed，也就不会有行为，就没有结果了，那么也就不造业了
enlightment==mirror_Alaya==0
#方法2：中阴得度
if(中阴身把心识交给佛陀) # 宗喀巴大师就是中阴身成就
enlightment == Alaya == 0
#方法3：大圆满（也是属于本尊修法）--米拉日巴尊者和莲花生大士

#如来藏类似数据库的镜像，之前是把五蕴的东西存到里面，现在是数据库转换成了阿赖耶识的镜子，他虽然接受五蕴的东西进来，
# 但是本身不存储而是照见五蕴皆空。看虽然看到那些东西了，但是内心不生意念，而是用心识照见五蕴。是善恶同体

if(Alaya==mirror_Alaya) 道成肉身也是肉身道成。善恶一体。
englightment==1

#自我强化学习=》有可能走向自我净化也有可能走向自我恶化  evil应该是一边净化心灵但同时享受身体乐趣的人。希特勒就是
# def build_q_table(n_states, actions):
#     table = pd.DataFrame(
#         np.zeros((n_states, len(actions))),     # q_table initial values全0初始化
#         columns=actions,    # actions's name
#     )
#     # print(table)    # show table
#     return table
#
# N_STATES = close[n]   # 第一次到close的bar的距离，比如可以从第一根k线开始
# ACTIONS = ['good', 'evil']     # available actions
# EPSILON = 0.6   # greedy police就是说90%的时候选择最优的动作，10%选随机的动作
# ALPHA = 0.1     # learning rate学习率  lr=F(Talent，Effort)
# GAMMA = 0.9    # discount factor未来奖励的衰减度
# MAX_EPISODES = 999999   # maximum episodes总共玩多少回合
# FRESH_TIME = 0.3    # fresh time for one move每走一步花多长时间

#heart-mind 心识  肉体 body

#智慧来自哪里？佛教的智慧来自空性，对镜

#慈悲来自哪里？
