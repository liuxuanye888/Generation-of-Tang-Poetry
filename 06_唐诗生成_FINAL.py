# -- encoding:utf-8 --


import os
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')


tf.app.flags.DEFINE_bool(
    'training', False, '定义是否训练，True为训练，False为预测'
)
FLAGS = tf.app.flags.FLAGS

"""
一、输入占位符的机制改进。 32个占位符  ---->  1个占位符。
二、使用one-hot 作为输入获得嵌入的输出 ---->  使用嵌入矩阵查找表。
三、预测： 取top 1 的值  ----> 取top 5的值。
          做藏头诗预测   的改进。
"""

class Tensor(object):
    def __init__(self, number_time_steps, num_units, vocab_size, embed_size, learning_rate):
        """
        构建模型图
        """
        self.learning_rate = learning_rate
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope("Net", initializer=tf.random_normal_initializer(0.0, 0.1)):
            batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            # 1. 定义RNN隐藏层
            cell = tf.nn.rnn_cell.MultiRNNCell(cells=[
                tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units),
                tf.nn.rnn_cell.GRUCell(num_units=num_units)
            ])
            cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=self.keep_prob)

            # 2. 获取得到初始化的状态信息
            state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            # 3. 对于每个时刻遍历，进行数据的输入操作
            self.rnn_outputs = []
            self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, number_time_steps])
            inputs = tf.split(self.inputs, num_or_size_splits=number_time_steps, axis=1)
            with tf.variable_scope("embeding", reuse=tf.AUTO_REUSE):
                embed_weight = tf.get_variable(name='w', shape=[vocab_size, embed_size])

            for time in range(number_time_steps):
                with tf.variable_scope("rnn_{}".format(time)):
                    # a. 定义当前时刻的输入
                    input_x = tf.nn.embedding_lookup(embed_weight, inputs[time])
                    input_x = tf.squeeze(input_x, axis=1)
                    # c. 调用RNN获取输出值和隐状态
                    output, state = cell(input_x, state)
                    # d. 将相关变量保存
                    self.rnn_outputs.append(output)

            losses = []
            self.y_predicts = []
            self.y_preds = []
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, number_time_steps])
            targets = tf.split(self.targets, num_or_size_splits=number_time_steps, axis=1)
            # b. 初始化全连接的参数
            with tf.variable_scope("fc_", reuse=tf.AUTO_REUSE):
                w = tf.get_variable(name='w', shape=[num_units, vocab_size])
                b = tf.get_variable(name='b', shape=[vocab_size])

            for time in range(number_time_steps):
                with tf.variable_scope('FC_{}'.format(time)):
                    # a. 获取对应时刻的输出
                    output = self.rnn_outputs[time]
                    logits = tf.add(tf.matmul(output, w), b)
                    # c. 获取预测值
                    y_predict = tf.argmax(logits, axis=1)
                    self.y_predicts.append(y_predict)

                    y_pred = tf.nn.softmax(logits)
                    self.y_preds.append(y_pred)

                    # d. 计算当前的损失函数
                    tmp_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.reshape(targets[time], shape=[-1]),
                        logits=logits))
                    losses.append(tmp_loss)
            self.loss = tf.reduce_mean(losses)
        with tf.variable_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(loss=self.loss, global_step=self.global_step)

        # 设置变量
        self.batch_size = batch_size


def chinese_to_index(text):
    result = []
    # https://www.qqxiuzi.cn/zh/hanzi-gb2312-bianma.php
    bs = text.encode("gb2312")
    num = len(bs)
    i = 0
    while i < num:
        b = bs[i]
        # 如果取值小于160，表示单个字符
        if b <= 160:
            result.append(b)
        else:
            # 计算区位码
            block = b - 160
            if block >= 16:
                # 因为10~15区为空, 所以不需要考虑
                block -= 6
            # 计算在当前区前有多有汉字
            block -= 1
            # 计算当前汉字在当前区位中的位置码
            i += 1
            b2 = bs[i] - 160 - 1
            # 基于区位码的信息计算出一个数字
            result.append(161 + block * 94 + b2)
        i += 1
    return result


def index_to_chinese(index_list):
    result = ''
    for index in index_list:
        if index <= 160:
            result += chr(index)
        else:
            index = index - 161
            block = int(index / 94) + 1
            if block >= 10:
                block += 6
            block += 160
            location = int(index % 94) + 1 + 160
            result += str(bytes([block, location]), encoding='gb2312')
    return result


def read_poems(path='../qts_7X4.txt'):
    """
    Load Data from File
    :param file_path:
    :return:
    """
    result = []
    error = 0
    with open(path, mode='r', encoding='utf-8') as reader:
        for line in reader:
            # 对每行数据进行处理
            line = line.strip()
            length = len(line)
            try:
                if length == 32:
                    index = chinese_to_index(line)
                    result.append(index)
                else:
                    error += 1
            except:
                error += 1
    print("成功获取诗歌:{}, 错误:{}".format(len(result), error))
    return result


def fetch_samples(path='../qts_7X4.txt'):
    """
    基于原始数据构建X和Y
    :return:
    """
    total_samples = 0
    X = read_poems(path)
    Y = []
    for xi in X:
        total_samples += 1
        # 使用前一个字预测下一个字
        yi = xi[1:]
        yi.append(10)
        # 添加到列表中
        Y.append(yi)
    # 将其转换为numpy数组的形式
    X = np.asarray(X)
    Y = np.asarray(Y)
    return total_samples, X, Y


def create_dir_with_not_exits(dir_path):
    """
    如果文件的文件夹路径不存在，直接创建
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def train(train_data_path, checkpoint_dir, max_epoch=100, batch_size=16, num_units=256):
    create_dir_with_not_exits(checkpoint_dir)
    with tf.Graph().as_default():
        # 一、构建网络
        tensor = Tensor(number_time_steps=32, num_units=num_units,
                        vocab_size=8000, embed_size=300, learning_rate=0.001)

        # 三、执行图的运行
        # 1. 构造持久化对象
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            # 做模型的恢复操作
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("进行模型恢复操作...")
                # 恢复模型
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                # 2. 模型初始化或者模型恢复
                sess.run(tf.global_variables_initializer())

            # 3. 迭代数据的获取，然后进行模型训练
            total_samples, X, Y = fetch_samples(train_data_path)
            total_batch = total_samples // batch_size
            times = 0
            random_index = np.random.permutation(total_samples)
            for epoch in range(max_epoch):
                # 获取当前批次对应的训练数据
                start_idx = times * batch_size
                end_idx = start_idx + batch_size
                idx = random_index[start_idx:end_idx]
                train_x = X[idx]
                train_y = Y[idx]
                # 构建数据输入对象
                feed_dict = {
                    tensor.batch_size: batch_size,
                    tensor.keep_prob: 0.7,
                    tensor.inputs: train_x,
                    tensor.targets: train_y
                }

                # 模型训练
                _, _loss = sess.run([tensor.train_op, tensor.loss], feed_dict=feed_dict)

                # 日志打印
                if epoch % 20 == 0:
                    print("Epoch:{}, Loss:{:.5f}".format(epoch, _loss))

                if (epoch+1) % 500 == 0:
                    save_path = os.path.join(checkpoint_dir, 'model.ckpt')
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    saver.save(sess, save_path, global_step=epoch)
                    print('Model saved in epoch:{}'.format(epoch))

                # 更新样本顺序
                times += 1
                if times == total_batch:
                    times = 0
                    random_index = np.random.permutation(total_samples)


def pick_top_n(preds, vocab_size, top_n=5):
    """
        随机从前n个概率最大的值中选取一个值作为预测值。
        :param preds:        预测的概率
        :param vocab_size:   单词表的大小
        :param top_n:        选取前n个概率最大的
        :return:
    """
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def predict_new(first, checkpoint_dir, num_units=256, vacab_size=8000):
    """
    基于第一个汉字生成一个唐诗
    :param first:
    :return:
    """
    if len(first) == 1:
        result = []
        with tf.Graph().as_default():
            # 一、构建网络
            tensor = Tensor(number_time_steps=32, num_units=num_units,
                            vocab_size=vacab_size, embed_size=300, learning_rate=0.001)

            # 二、模型加载及预测
            with tf.Session() as sess:
                # a. 加载模型
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reload Model...")
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("加载失败...")

                # b. 做一个预测
                first_idx = chinese_to_index(first)[0]
                input_list = np.zeros(shape=[1, 32])    # 因为我们模型中定义的占位符是32个时刻，所以先将后面31个输入设置为0，后面预测出来值后逐个替换掉。
                input_list[:, 0] = first_idx

                result.append(first_idx)
                feed_dict = {
                    tensor.batch_size: 1,
                    tensor.inputs: input_list,
                    tensor.keep_prob: 1.0
                }
                for time in range(1, 32):
                    # 获取time-1时刻对应的预测概率值
                    predict = sess.run(tensor.y_preds[time - 1],
                                       feed_dict=feed_dict)
                    c = pick_top_n(predict, vocab_size=vacab_size, top_n=5)
                    # 设置time时刻对应的输入
                    input_list[:, time] = c
                    result.append(c)
                # print("预测值为:{}".format(result))
                print("预测的唐诗为:\n{}".format(index_to_chinese(result)))
    else:
        # 做一首藏头唐诗
        result = []
        with tf.Graph().as_default():
            # 一、构建网络
            tensor = Tensor(number_time_steps=32, num_units=num_units,
                            vocab_size=8000, embed_size=300, learning_rate=0.001)

            # 二、模型加载及预测
            with tf.Session() as sess:
                # a. 加载模型
                saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    print("Reload Model...")
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("加载失败...")

                # b. 做一个预测
                first_idx = chinese_to_index(first)
                input_list = np.zeros(shape=[1, 32], dtype=int)  # 因为我们模型中定义的占位符是32个时刻，所以先将后面31个输入设置为0，后面预测出来值后逐个替换掉。
                for idx, word in zip([0, 8, 16, 24], first_idx):
                    input_list[:, idx] = word

                result.append(first_idx[0])
                feed_dict = {
                    tensor.batch_size: 1,
                    tensor.inputs: input_list,
                    tensor.keep_prob: 1.0
                }
                for time in range(1, 32):
                    # todo 如果是这几个时刻，就直接用输入的藏头字。
                    if time in [8, 16, 24]:
                        result.append(input_list[:, time][0])
                        continue
                    else:
                        # 获取time-1时刻对应的预测概率值
                        predict = sess.run(tensor.y_preds[time - 1],
                                           feed_dict=feed_dict)
                        c = pick_top_n(predict, vocab_size=vacab_size, top_n=5)
                        # 设置time时刻对应的输入
                        input_list[:, time] = c
                        result.append(c)
                # print("预测值为:{}".format(result))
                print("预测的唐诗为:\n{}".format(index_to_chinese(result)))


def main(_):
    opt = 1  # 1代表用pycharm跑 2 # 用的控制台界面跑
    if opt == 1:
        file_path = './qts_7X4.txt'
        checkpoint_dir = './model_embed06'
    elif opt == 2:
        file_path = r'F:\python\深度学习\RNN循环神经网络\唐诗生成\qts_7X4.txt'
        checkpoint_dir = r'F:\python\深度学习\RNN循环神经网络\唐诗生成\model_embed06'

    print(FLAGS.training)
    if FLAGS.training:
        train(train_data_path=file_path, checkpoint_dir=checkpoint_dir, max_epoch=3000)
    else:
        predict_new('天', checkpoint_dir=checkpoint_dir)
        predict_new('春', checkpoint_dir=checkpoint_dir)
        predict_new('我爱中国', checkpoint_dir=checkpoint_dir)
        predict_new('春夏秋冬', checkpoint_dir=checkpoint_dir)


if __name__ == '__main__':
    tf.app.run()
