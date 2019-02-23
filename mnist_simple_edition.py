# author: Logan Qin
# email: 65207907@qq.com
# time: 2019/2/23 17:13

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载数值
# 网络方式直接下载获取MNIST
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 或者本地直接获取MNIST
mnist = input_data.read_data_sets("D:\Python\dataSet", one_hot=True)

# 设置每次训练的批次
batch_size = 100

# 计算样本数据分成了多少个批次
n_batch = mnist.train.num_examples


x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义假设函数，其中需要注意matmul(x, w)中x和w参数的位置
hyper_function = tf.matmul(x, w) + b

# 使用softmax网络
prediction = tf.nn.softmax(hyper_function)

# 定义损失函数——二次代价函数
# loss  = tf.reduce_mean(tf.square(y - prediction))

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
    for epoch in range(4):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})

        accuracy_each_epoch = sess.run(accuracy, feed_dict= {x:mnist.test.images, y:mnist.test.labels})
        print('epoch' + str(epoch) + ',当前训练轮次的预测准确率为：' + str(accuracy_each_epoch))
