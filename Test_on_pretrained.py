numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

import numpy as np
import tensorflow as tf


wordsList = [x.decode('utf-8').lower() for x in np.load('training_data/wordsList.npy').tolist()]
wordVectors = np.load('training_data/wordVectors.npy')

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

#----------------------loading the tested data----------------------#
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
#saver = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-90000.meta')
#saver.restore(sess, tf.train.latest_checkpoint('models/./'))

new_saver_file = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-90000.meta')
#print(tf.train.latest_checkpoint('models/'))
saver.restore(sess, tf.train.latest_checkpoint('models'))


import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix

inputText = "Quality is not an act, it is a habit."
inputMatrix = getSentenceMatrix(inputText)

predictedSentiment = sess.run(prediction, feed_dict={input_data: inputMatrix})[0]

print(predictedSentiment)
if predictedSentiment[0] > predictedSentiment[1]:
    print("Positive Sentiment: ", predictedSentiment[0])
else:
    print("Negative Sentiment: ", predictedSentiment[1])

# secondInputText = "That movie was the best one I have ever seen."
# secondInputMatrix = getSentenceMatrix(secondInputText)
#
# predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]
#
# if predictedSentiment[0] > predictedSentiment[1]:
#     print("Positive Sentiment: ", predictedSentiment[0])
# else:
#     print("Negative Sentiment: ", predictedSentiment[1])