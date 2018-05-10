import gzip
import pickle

import numpy as np
import tensorflow as tf

recall = 1
batch_size = 5000
path = "./word2vec/gensim_tmp/gensim.word2vec.embedd.pkl.gz"  # 0.24765804678296965 4415/17827
path = "./word2vec/word2vec_tmp/nce.embedd.pkl.gz"
path = "./word2vec/word2vec_tmp/sm.embedd.pkl.gz"
path = "./glove/glove_tmp/glove.embedd.pkl.gz"

with gzip.open(path, 'rb') as f:
    dictionary, embeddings_from_file = pickle.load(f)

print(embeddings_from_file.shape, " just loaded")

vocabulary_size = embeddings_from_file.shape[0]
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

graph = tf.Graph()
with graph.as_default():
    with tf.variable_scope('eval'):
        """Build the eval graph."""
        embeddings = tf.constant(embeddings_from_file, tf.float32)
        normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(normalized_embeddings, analogy_a)  # a's embs
        b_emb = tf.gather(normalized_embeddings, analogy_b)  # b's embs
        c_emb = tf.gather(normalized_embeddings, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)  # because a,b,c,d is assumed to be a - b = c -d
        normalized_target = tf.nn.l2_normalize(target, 1)
        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(normalized_target, normalized_embeddings, transpose_b=True)

        # For each question (row in dist), find the top recall words.
        _, analogy_pred_recall_plus_three_idx = tf.nn.top_k(dist, recall + 3)  # 3 for a b c words

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(normalized_embeddings, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, normalized_embeddings, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist, min(1000, vocabulary_size))

eval_path = './eval_data/questions-words.txt'


def read_analogies():
    """Reads through the analogy question file.

    Returns:
      questions: a [n, 4] numpy array containing the analogy question's
                 word ids.
      questions_skipped: questions skipped due to unknown words.
    """
    questions = []
    questions_skipped = 0
    question_classes = []
    count_per_class = 0
    count_total = 0
    with open(eval_path, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                if question_classes:
                    question_classes[-1] = (question_classes[-1][0], count_per_class)
                    count_total += count_per_class
                question_classes.append((line, 0))
                count_per_class = 0
            else:
                words = line.strip().lower().split(" ")
                ids = [dictionary.get(w.strip()) for w in words]

                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                    questions.append(np.array(ids))

                count_per_class += 1

    question_classes[-1] = (question_classes[-1][0], count_per_class)
    count_total += count_per_class

    print("Eval analogy file: ", eval_path)
    print("Questions: ", len(questions))
    assert len(questions) + questions_skipped == count_total
    print("Skipped: ", questions_skipped)
    print(question_classes)
    return np.array(questions, dtype=np.int32)


def predict_recall_plus_three_idx(analogies, sess):
    """Predict the top 4 answers for analogy questions."""
    idx, a, b = sess.run([analogy_pred_recall_plus_three_idx, normalized_target, dist], {
        analogy_a: analogies[:, 0],
        analogy_b: analogies[:, 1],
        analogy_c: analogies[:, 2]
    })
    return idx


def my_tf_eval(sess):
    """Evaluate analogy questions and reports accuracy."""
    # from word_vectors.word2vec.mygensim import my_gs_eval
    # predictions1 = my_gs_eval()
    # # How many questions we get right at precision@1.
    # predictions2 = []
    correct = 0
    analogy_questions = read_analogies()
    try:
        total = analogy_questions.shape[0]  # number of question
    except AttributeError as e:
        raise AttributeError("Need to read analogy questions.")

    start = 0
    while start < total:
        limit = start + batch_size
        batch = analogy_questions[start:limit, :]
        idx = predict_recall_plus_three_idx(batch, sess)
        start = limit

        for question in range(batch.shape[0]):  # batch_size or less
            for index in idx[question]:
                if index == batch[question, 3]:
                    # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                    correct += 1
                elif index not in batch[question, :3]:
                    break  # not in  precision@1.

    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))


def analogy_for_word(w0, w1, w2, sess):
    """Predict word w3 as in w0:w1 vs w2:w3."""
    wid = np.array([[dictionary.get(w, 0) for w in [w0, w1, w2]]])
    top_four_word_indexes = predict_recall_plus_three_idx(wid, sess)

    for c in [reverse_dictionary[word_index] for word_index in top_four_word_indexes[0, :]]:
        if c not in [w0, w1, w2]:
            print(c)
            return
    print("unknown")


def nearby(words, sess, num=20):
    """Prints out nearby words given a list of words."""
    ids = np.array([dictionary.get(x, 0) for x in words])

    neig_cos_distances, idx = sess.run([nearby_val, nearby_idx], {nearby_word: ids})

    for i in range(len(words)):
        print("\n%s\n=====================================" % (words[i]))
        for (neighbor_id, neig_cos_distance) in zip(idx[i, :num], neig_cos_distances[i, :num]):
            print("%-20s %6.4f" % (reverse_dictionary[neighbor_id], neig_cos_distance))


with tf.Session(graph=graph) as session:
    # my_tf_eval(session)
    nearby(["one", "fish", "cairo", "human", "mohammed", "time", "standards", "program", "machine", "intelligence"], session, 10)
