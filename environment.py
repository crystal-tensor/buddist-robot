import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import math
import keras
import pymysql

firstmind = outputLayer(ar/reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 初心
rein_conscions = outputLayer(emotion/reward, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 结果和初心的距离
#acction = outputLayer(reward/moveloss, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)   # 行为强化
output_logic = outputLayer(firstmind/ar, input_size, 1,  n_layer=1, activation_function=tf.nn.tanh)   # 结果和推理的距离
result_alpha = outputLayer(ar, input_size, 1,  n_layer=1, activation_function=tf.nn.tanh)   # 分类和推理的距离  任何的分类都是一种方法
position = addLayer(output_logic/result_alpha, input_size, 1, n_layer=1, activation_function=tf.nn.tanh)