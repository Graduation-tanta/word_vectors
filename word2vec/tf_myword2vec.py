# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
# %matplotlib inline
from __future__ import print_function

import collections
import math
import random
import sys
from random import shuffle

import numpy as np
import tensorflow as tf
from six.moves import range

sys.stdout = open('sm', 'w')


# url = 'http://mattmahoney.net/dc/text8.zip'


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with open(filename, 'r') as f:
        data = f.read()

    return data.split(' ')


# words = read_data('D:\preferences\jetBrains\pycharm\\udacity\\text8.data')
words = read_data('./text8')

# words = words[0:500000]
print('Data size %d' % len(words))
print(words[0:15])
vocabulary_size = 71290


# extract features by mapping each word to number and vice versa
# make count array of the most common words
# make a dictionary for word to num             word -> rank
# make reverse dictionary       word <- rank
# map each word to its number(rank or 0 for UNK) as list of words
def build_dataset(words):
    count = [['UNK', -1]]  # frequency of the most common 50000 word
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()  # map common words to >> number according to frequency 0 for UNK 1 for THE
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()  # same as words list but of numbers corresponding to each word
    unk_count = 0
    for word in words:
        if word in dictionary:  # is it a common word ?
            index = dictionary[word]  # it's rank
        else:
            index = 0  # UNK is mapped to 0
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count  # ['UNK', -1] => ['UNK', unk_count]
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

data_index = 0


# Function to generate a training batch for the skip-gram model. one to many predictions
# span is [ skip_window target skip_window ]
# batch = [num_skips1 num_skips2 num_skips3 num_skips4 ...]
def generate_batch(batch_size, num_skips_size, skip_window):
    global data_index
    assert batch_size % num_skips_size == 0  # assertion is sanity check for development   error if false
    assert num_skips_size <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # double ended queue holds the current memory in consideration
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for skip_index in range(batch_size // num_skips_size):
        context_word = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]

        for j in range(num_skips_size):  # num_skips predicted outputs for a given word
            while context_word in targets_to_avoid:
                context_word = random.randint(0, span - 1)
            targets_to_avoid.append(context_word)
            batch[skip_index * num_skips_size + j] = buffer[skip_window]
            labels[skip_index * num_skips_size + j, 0] = buffer[context_word]

        buffer.append(data[data_index])

        if (data_index + 1) % len(data) < data_index + 1:
            return "koko"
        data_index = (data_index + 1) % len(data)

    return batch, labels


# gets the first 8 words mapped from numbers
print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips_size=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

data_index = 0
batch_size = 200
embedding_size = 200  # Dimension of the embedding vector.
skip_window = 5  # How many words to consider left and right.
num_skips = 10  # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 5  # Number of negative examples to sample.

data_length = len(data)
learning_rate = 0.025

num_epochs = 16
steps_per_epoch = data_length // (batch_size // num_skips + 2 * skip_window + 1)
graph = tf.Graph()

with graph.as_default():
    with tf.variable_scope('training'):
        # Input data.
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # classes=vocabulary_size
        # Variables.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name='embeddings')

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)), name='weights')
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name='biases')  # classes

        # Model.
        # Look up embeddings for inputs.
        embed = tf.nn.embedding_lookup(embeddings, train_dataset)
        print(embed.get_shape().as_list())  # samples * features
        # Compute the softmax loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=nce_weights,  # or nce_loss
                                                         biases=nce_biases,
                                                         inputs=embed,
                                                         labels=train_labels,
                                                         num_sampled=num_sampled,
                                                         num_classes=vocabulary_size))

        # nce loss
        # num_sampled  The number of classes to randomly sample per batch which are negative ones to update .
        # we only update (1 + 64) * 128 weights instead of all outputs

        # Optimizer.
        # Note: The optimizer will optimize the softmax_weights AND the embeddings.
        # This is because the embeddings are defined as a variable quantity and the
        # optimizer's `minimize` method will by default modify all variable quantities
        # that contribute to the tensor it is passed.
        # See docs on `tf.train.Optimizer.minimize()` for more details.
        global_step = tf.Variable(0)  # count the number of steps taken.
        lr = learning_rate * tf.maximum(0.0001, 1.0 - tf.cast(global_step, tf.float32) / (num_epochs * data_length // (batch_size // num_skips)))
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_step = optimizer.minimize(loss, global_step=global_step, gate_gradients=optimizer.GATE_NONE)

        # Compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance:
        # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))  # keep_dims means (50000,1) instead of (50000,)
        # print('norm ', normalized_embeddings.get_shape().as_list())
        # normalized_embeddings = embeddings / norm

        normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)
        print('normalized_embeddings ', normalized_embeddings.get_shape().as_list())

        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)  # valid_size * embedding
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))  # valid_size * embedding MATMUL  embedding* vocab
        print('similarity ', similarity.get_shape().as_list())  # valid_size * vocab

        _, top_indexes = tf.nn.top_k(similarity, 8)

print(steps_per_epoch)

batchs = []
import time

start = time.clock()
for _ in range(steps_per_epoch):
    batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    batchs.append((batch_data, batch_labels))

print("data batches generated", time.clock() - start)

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0

    for epoch in range(num_epochs):
        start = time.clock()
        shuffle(batchs)
        for step in range(steps_per_epoch):
            batch_data, batch_labels = batchs[step]
            feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
            _, l, _lr = session.run([train_step, loss, lr], feed_dict=feed_dict)
            average_loss += l

            if step != 0 and step % (steps_per_epoch // 50) == 0:
                average_loss = average_loss / (steps_per_epoch // 50)
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d/%d: %f as learning rate is %f' % (step, steps_per_epoch, average_loss, _lr))
                average_loss = 0

            sys.stdout.flush()

        np_top_indexes = top_indexes.eval()
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            log = 'Nearest to %s:' % valid_word
            for k in np_top_indexes[i][1:]:
                close_word = reverse_dictionary[k]
                log = '%s %s,' % (log, close_word)
            print(log)

        print("epoch time", time.clock() - start)
        sys.stdout.flush()
    final_embeddings = normalized_embeddings.eval()

    import pickle
    import gzip

    f = gzip.open("sm.embedd.pkl.gz", "w")
    pickle.dump((dictionary, final_embeddings), f)
    f.close()
