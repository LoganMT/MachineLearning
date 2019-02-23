# author: Logan Qin
# email: 65207907@qq.com
# time: 2019/2/23 18:28

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数值
# 或者本地直接获取MNIST
mnist = input_data.read_data_sets("D:\Python\dataSet", one_hot=True)

# 设置每次训练的批次
batch_size = 100

# 计算样本数据分成了多少个批次
n_batch = mnist.train.num_examples


x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])

# dropout的比例
keep_alive_percent = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
b1 = tf.Variable(tf.zeros([100]) + 0.15 )
# 定义假设函数，其中需要注意matmul(x, w)中x和w参数的位置
hyper_function_1 = tf.matmul(x, w1) + b1
# 使用tanh网络
layer_1 = tf.nn.tanh(hyper_function_1)
# 使用dropuot
layer_1_drop = tf.nn.dropout(layer_1, keep_alive_percent)


w2 = tf.Variable(tf.truncated_normal([100, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([100]) + 0.15 )
hyper_function_2 = tf.matmul(layer_1_drop, w2) + b2
layer_2 = tf.nn.tanh(hyper_function_2)
layer_2_drop = tf.nn.dropout(layer_1, keep_alive_percent)


w3 = tf.Variable(tf.truncated_normal([100, 100], stddev=0.1))
b3 = tf.Variable(tf.zeros([100]) + 0.15 )
hyper_function_3 = tf.matmul(layer_2_drop, w3) + b3
layer_3 = tf.nn.tanh(hyper_function_3)
layer_3_drop = tf.nn.dropout(layer_3, keep_alive_percent)

w4 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.15 )
prediction = tf.nn.softmax(tf.matmul(layer_3_drop, w4) + b4)



# 或者（优化）使用交叉熵,记得求平均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法寻找最优的解
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 初始化定义的变量
init = tf.global_variables_initializer()

# 返回softmax十个类别中最大概率的位置
position_real = tf.argmax(y, 1)
position_prediction = tf.argmax(prediction, 1)

# 比较分类的正确性
corret_prediction = tf.equal(position_real , position_prediction)

# 求预测的准确率
accuracy = tf.reduce_mean(tf.cast(corret_prediction, tf.float32))


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_alive_percent:0.95})

        accuracy_each_epoch = sess.run(accuracy,
                                       feed_dict= {x:mnist.test.images,
                                                   y:mnist.test.labels,
                                                   keep_alive_percent:0.95})
        print('epoch' + str(epoch)+ ',当前训练轮次的预测准确率为：' + str(accuracy_each_epoch))
