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

# 将平铺的图像转换为三维矩阵,黑白minist数据集通道数为1
# image shape (batch,28,28,1)
# 使用tf.reshape
image = tf.reshape(x, [-1, 28, 28, 1])

#直接调用封装好的tf.layers.conv2d，或者是使用tf.nn.conv2d需要自己定义卷积核
#查阅相关相关资料完成参数的填写
#通道数为16，卷积核大小为5x5，步长为1，填充方式为same
# 输出图像大小-> (28, 28, 16)
conv1 = tf.layers.conv2d(inputs = image, filters = 16, kernel_size = 5, padding = "same")
#直接调用分装好的池化tf.layers.max_pooling2d 或者使用tf.nn.max_pool
#步长为2，池化窗口为2x2
# 输出图像大小-> (14, 14, 16)
pool1 = tf.layers.max_pooling2d(inputs = conv1, strides = 2, pool_size = (2,2))

#第二次卷积，通道数为32，卷积核大小为5x5，步长为1，填充方式为same
# 输出图像大小-> (14, 14, 32)
conv2 = tf.layers.conv2d(inputs = pool1, filters = 32, kernel_size = 5, padding = "same")
#步长为2，池化窗口为2x2
# 输出图像大小-> (7, 7, 32)
pool2 = tf.layers.max_pooling2d(inputs = conv2, strides = 2, pool_size = (2,2))

# 通过tf.reshape将图像平铺-> (7*7*32)
flat = tf.reshape(pool2, [-1, 7*7*32])
# 通过封装好的全连接层tf.layers.dense获得预测的标签分布，共10分类
pred_y = tf.layers.dense(inputs = flat, units = 10, activation = tf.nn.relu)

#通过tf.nn.softmax_cross_entropy_with_logits_v2定义交叉熵损失函数
loss= tf.nn.softmax_cross_entropy_with_logits_v2(labels = gt_y, logits = pred_y)
#通过tf.reduce_mean将多个样本的损失求均值
loss= tf.reduce_mean(loss)
#通过tf.train.GradientDescentOptimizer(lr).minimize(loss)定义学习率和梯度更新的方法
optim = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化tensorflow
with tf.Session() as sess:
    #随机初始化参数
    tf.global_variables_initializer().run()
    print('start training...')

    # training
    for i in range(500):
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