import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
input_size = 784
output_size = 10
BATCH_SIZE = 50
LR = 0.001              # learning rate

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0])); plt.show()

xs = tf.placeholder(tf.float32, [None, 784]) / 255.
image = tf.reshape(xs, [-1, 28, 28, 1])              # (batch, height, width, channel)
ys = tf.placeholder(tf.int32, [None, 10]) # input y

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

# emotion = addLayer(ys, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)  # 情绪取决于移动止损的距离 值在-1到1之间
# gb = addLayer(emotion/test_x, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)  # 好和坏的初级判断或者叫初级推理，取决于收益和风险
# ar = addLayer(test_x/emotion, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)   # 取舍逻辑，推理是否接受， 值在-1到1之间 ，值越接近0代表接受的程度越大
# firstmind = addLayer(ar/test_x, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)   # 初心
# #alpha_neural = adLayer(alpha, input_size, 1, activity_function=tf.nn.tanh)
#
# #环境评估
# output_logic = addLayer(firstmind/ar, input_size, output_size,  n_layer=1, activation_function=tf.nn.tanh)   # 结果和推理的距离
# result_alpha = addLayer(ar, input_size, output_size,  n_layer=1, activation_function=tf.nn.tanh)   # 分类和推理的距离  任何的分类都是一种方法
# position = addLayer(output_logic/result_alpha, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)  #对位置的判断决定了你认识环境的高低，此处位置越稳定说明你对环境的认识越到位。
# #self：
# reinforceconscious = addLayer(firstmind/(test_x-emotion), input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)
# unconscious=addLayer(firstmind+reinforceconscious, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)  #love+gene_defect   #无意识来自爱和基因缺陷
# wisdow=addLayer(unconscious/reinforceconscious, input_size, output_size, n_layer=1, activation_function=tf.nn.tanh)
# CNN
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2,
)           # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
output = tf.layers.dense(flat, 10)              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(ys, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
def plot_with_labels(lowDWeights, labels):
    plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()
for step in range(60):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {xs: b_x, ys: b_y})
    if step % 10 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat], {xs: test_x, ys: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
            low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
            labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, {xs: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')

if int((tf.__version__).split('.')[1]) < 12 and int(
            (tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
            writer = tf.train.SummaryWriter('logs/', sess.graph)
else:  # tensorflow version >= 0.12
            writer = tf.summary.FileWriter("logs/", sess.graph)
#tensorboard --logdir=logs