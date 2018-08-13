from __future__ import print_function
import pandas as pd
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from sklearn  import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import pymysql

conn = pymysql.Connect(host='127.0.0.1',port=3306,user='root',passwd='',db='stockcn',charset='utf8')
cursor = conn.cursor()
#sql = "select * from stock_000001"
#print("cursor.excute:",cursor.rowcount)

cursor.execute("select high_price,low_price from day_price where symbol='600000' ")
#data = cursor.fetchone()
data = cursor.fetchall()
cursor.execute("select low_price,low_price from day_price where symbol='600000' ")
#data = cursor.fetchone()
data2 = cursor.fetchall()

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

alpha = preprocessing.minmax_scale(data,feature_range=(-1, 1))   #alpha的因变量
alpha2 = preprocessing.minmax_scale(data2,feature_range=(-1, 1))   #alpha的因变量
print(alpha2)
#x2 = tf.placeholder(dtype=tf.float32,shapes=[None])
neww = addLayer(alpha+alpha2, 1, 1, n_layer=1, activation_function=tf.nn.tanh)
print(alpha)