#%tensorflow_version 1.x
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print(tf.__version__)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#通过tf.placeholder定义图片的输入和标签,
# 图片为28x28的一维向量，标签为10分类的one hot
#imgs shape: (n,784) labels shape: (n,10)
x = tf.placeholder(tf.float32, [None,784])
gt_y = tf.placeholder(tf.float32, [None, 10])
#通过tf.Variable定义mlp的wx+b中参数w和b
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#通过tf.matmul计算矩阵乘法
pred_y =tf.matmul(x, W)+b

#通过tf.nn.softmax_cross_entropy_with_logits_v2定义交叉熵损失函数
loss= tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_y, logits = pred_y)
#通过tf.reduce_mean将多个样本的损失求均值
loss= tf.reduce_mean(loss)
#通过tf.train.GradientDescentOptimizer(lr).minimize(loss)定义学习率和梯度更新的方法
optim = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

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
        # 参考sess.run运行tensorflow，包括执行的占位符和输入数据，返回执行占位符的列表
        _, step_loss = sess.run([optim,-tf.reduce_sum(gt_y * tf.log(pred_y+1e-10))], feed_dict={x: batch_xs, gt_y: batch_ys})
        if i%100==0:
          print('step {} loss: {:.2f}'.format(i,step_loss))

    # test
    # tf.equal逐个判断两个矩阵中元素是否相等，tf.argmax求一个矩阵中最大的下标值
    correct_prediction = tf.equal(tf.argmax(pred_y, 1), tf.argmax(gt_y, 1))
    #计算精度
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #输入精度的占位符和喂给测试数据
    acc_res=sess.run(accuracy, feed_dict={x: mnist.test.images, gt_y: mnist.test.labels}) # 0.9194
    print(acc_res)
