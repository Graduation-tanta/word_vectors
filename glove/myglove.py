# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
# %matplotlib inline
from __future__ import print_function

import collections
import random
import sys
import time
from collections import defaultdict
from random import shuffle

import numpy as np
import tensorflow as tf
from six.moves import range

sys.stdout = open('glove', 'w')


# url = 'http://mattmahoney.net/dc/text8.zip'


min_occurrences = 25
count_max = 100
scaling_factor = 3 / 4
batch_size = 512
embedding_size = 150  # Dimension of the embedding vector.
window_size = 5  # How many words to consider left and right.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 5000  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))

learning_rate = 0.05
num_epochs = 50


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with open(filename, 'r') as f:
        data = f.read()

    return data.split(' ')


#words = read_data('D:\preferences\jetBrains\pycharm\\udacity\\text8.data')
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
    global vocabulary_size
    count = [['UNK', -1]]  # frequency of the most common 50000 word

    count.extend([(word, count) for word, count in collections.Counter(words).most_common(vocabulary_size - 1) if count >= min_occurrences])
    vocabulary_size = len(count)

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


# build the dataset
def generate_cooccurrences():
    co_occurrence_counts = defaultdict(float)
    for focal_index in range(window_size, len(data) - window_size):
        focal_word = data[focal_index]

        for dist, context_word in enumerate(data[focal_index - window_size: focal_index]):  # left to right
            co_occurrence_counts[(focal_word, context_word)] += (1 / (window_size - dist))

        for dist, context_word in enumerate(data[focal_index + 1:focal_index + 1 + window_size]):  # left to right
            co_occurrence_counts[(focal_word, context_word)] += (1 / (1 + dist))

    return co_occurrence_counts


co_occurrence_counts = list(generate_cooccurrences().items())
steps_per_epoch = len(co_occurrence_counts) // batch_size
data_index = 0

print(steps_per_epoch)


# Function to generate a training batch for the glove model.
# span is [ skip_window target skip_window ] on which co occurrence is defined
def generate_batch():
    global data_index
    focal = np.ndarray(shape=[batch_size], dtype=np.int32)
    context = np.ndarray(shape=[batch_size], dtype=np.int32)
    counts = np.ndarray(shape=[batch_size], dtype=np.float32)

    for ind, ((cen, con), cou) in enumerate(co_occurrence_counts[data_index:data_index + batch_size]):
        focal[ind], context[ind], counts[ind] = cen, con, cou
    data_index += batch_size
    return focal, context, counts


batchs = []

start = time.clock()
for _ in range(steps_per_epoch):
    focal, context, counts = generate_batch()
    batchs.append((focal, context, counts))

print("data batches generated", time.clock() - start)

graph = tf.Graph()


def device_for_node(n):
    if n.type == "MatMul":
        return "/gpu:0"
    else:
        return "/cpu:0"


with graph.as_default():
    with graph.device(device_for_node):
        with tf.variable_scope('training'):
            # system parameters
            # count_max = tf.constant([_count_max], dtype=tf.float32)
            # scaling_factor = tf.constant([_scaling_factor], dtype=tf.float32)

            # Input data place holders.
            focal_input = tf.placeholder(tf.int32, shape=[batch_size])
            context_input = tf.placeholder(tf.int32, shape=[batch_size])
            co_occurrence_count_input = tf.placeholder(tf.float32, shape=[batch_size])

            # validation dataset
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            # Variables.
            focal_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], 1.0, -1.0))
            context_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], 1.0, -1.0))
            focal_biases = tf.Variable(tf.random_uniform([vocabulary_size], 1.0, -1.0))
            context_biases = tf.Variable(tf.random_uniform([vocabulary_size], 1.0, -1.0))

            # Model.
            # Look up embeddings for inputs.
            focal_embedding = tf.gather(focal_embeddings, focal_input)
            context_embedding = tf.gather(context_embeddings, context_input)
            focal_bias = tf.gather(focal_biases, focal_input)
            context_bias = tf.gather(context_biases, context_input)

            weighting_factor = tf.minimum(1.0, tf.pow(tf.div(co_occurrence_count_input, count_max), scaling_factor))

            distance_expr = tf.square(tf.add_n([
                tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1),
                focal_bias,
                context_bias,
                tf.negative(tf.log(co_occurrence_count_input))
            ]))

            # Optimizer.
            total_loss = tf.reduce_sum(tf.multiply(weighting_factor, distance_expr))  # losses on batch
            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))  # keep_dims means (50000,1) instead of (50000,)
            # print('norm ', normalized_embeddings.get_shape().as_list())
            # normalized_embeddings = embeddings / norm
            normalized_embeddings = tf.nn.l2_normalize(tf.add(focal_embeddings, context_embeddings), 1)
            print('normalized_embeddings ', normalized_embeddings.get_shape().as_list())

            valid_embeddings = tf.gather(normalized_embeddings, valid_dataset)  # valid_size * embedding
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)  # valid_size * embedding MATMUL  embedding* vocab
            print('similarity ', similarity.get_shape().as_list())  # valid_size * vocab

            _, top_indexes = tf.nn.top_k(similarity, 8)

sys.stdout = open('glove', 'w')
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0

    for epoch in range(num_epochs):
        start = time.clock()
        shuffle(batchs)
        for step in range(steps_per_epoch):
            batch_focal_words, batch_context_words, batch_co_occurrences = batchs[step]
            feed_dict = {focal_input: batch_focal_words, context_input: batch_context_words, co_occurrence_count_input: batch_co_occurrences}
            _, l = session.run([optimizer, total_loss], feed_dict=feed_dict)
            average_loss += l

            if step != 0 and step % (steps_per_epoch // 50) == 0:
                average_loss = average_loss / (steps_per_epoch // 50)
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d/%d: %f ' % (step, steps_per_epoch, average_loss))
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

    f = gzip.open("glove.embedd.pkl.gz", "w")
    pickle.dump((dictionary, final_embeddings), f)
    f.close()
