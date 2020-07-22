#%tensorflow_version 1.x
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist.train.images.shape)  # (55000, 784)
# print(mnist.train.labels.shape)  # (55000, 10)
# print(mnist.test.images.shape)  # (10000, 784)
# print(mnist.test.labels.shape)  # (10000, 10)

# input x shape (batch,28*28)
x = tf.placeholder(tf.float32, [None, 784])
# input gt_y shape (batch,10)
gt_y = tf.placeholder(tf.float32, [None, 10])

# image reshape (batch, height, width)
#注意我们这里把一张图片抽象成序列化的数据，height表示序列化文本的长度,width表示每个单词的维度
image = tf.reshape(x, [-1, 28, 28])

# RNN
#首先构造一个循环神经网络的神经单元，他的隐藏状态和输出状态的维度相同设置为64
#使用tf.nn.rnn_cell.LSTMCell或者GRUCell
rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units = 200, use_peepholes = True, num_proj = 200)
#使用tf.nn.dynamic_rnn将循环神经网络单元构造成一个序列网络
#循环神经网络的初始化状态可以自选设置
#输出为两个张量，一个是循环神经网络的输出另一个是各个时间节点的状态
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(cell = rnn_cell,inputs = image, initial_state = None, dtype=tf.float32, time_major = False)
#将循环神经网络最后一个输出作为mlp的输入，输出10分类的预测分布
pred_y = tf.layers.dense(inputs = outputs[:,-1,:], units = 10)


#通过tf.nn.softmax_cross_entropy_with_logits_v2定义交叉熵损失函数
loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_y,logits=pred_y)
#通过tf.reduce_mean将多个样本的损失求均值
loss=tf.reduce_mean(loss)
#通过tf.train.GradientDescentOptimizer(lr).minimize(loss)定义学习率和梯度更新的方法
optim = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化tensorflow
with tf.Session() as sess:
    #随机初始化参数
    tf.global_variables_initializer().run()
    print('start training...')

    # training
    for i in range(1000):
        # 在mnist.train中取100个训练数据
        # batch_xs是形状为(100, 784)的图像数据，batch_ys是形如(100, 10)的实际标签
        # batch_xs, batch_ys对应着两个占位符x和y_
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 在Session中运行train_step，运行时要传入占位符的值
        #参考sess.run运行tensorflow，包括执行的占位符和输入数据，返回执行占位符的列表
        _,step_loss=sess.run([optim,loss], feed_dict={x: batch_xs, gt_y: batch_ys})
        if i%100==0:
          print('step {} loss: {:.2f}'.format(i,step_loss))

    # test
    # tf.equal逐个判断两个矩阵中元素是否相等，tf.argmax求一个矩阵中最大的下标值
    correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(gt_y, 1))
    #计算精度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #输入精度的占位符和喂给测试数据
    acc_res=sess.run(accuracy, feed_dict={x: mnist.test.images, gt_y: mnist.test.labels}) # 0.9185
    print(acc_res)
