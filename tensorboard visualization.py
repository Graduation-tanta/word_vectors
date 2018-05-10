import errno
import gzip
import os
import pickle

import tensorflow as tf


def build_vis_embed(embeddings, labels_dictionary, target_dir):
    target_dir += '/check'
    if not os.path.exists(target_dir):
        try:
            os.makedirs(target_dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # Create randomly initialized embedding weights which will be trained.
    tf.Variable(embeddings, name='word_embedding')

    saver = tf.train.Saver()
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        saver.save(session, target_dir)

    with open(target_dir + '/metadata.tsv', 'w') as f:
        for key, val in labels_dictionary.items():
            f.write('{}\n'.format(key))

    with open(target_dir + '/projector_config.pbtxt', 'w') as f:
        f.write("""embeddings {
      tensor_name: "word_embedding:0"
      metadata_path: "./metadata.tsv"
    }
    """)


################################################################################################################################################################################
with gzip.open('D:\preferences\jetBrains\pycharm\Projects Cocktail\word_vectors\word2vec\gensim_tmp\gensim.word2vec.embedd.pkl.gz', 'rb') as f:
    dictionary, embeddings_from_file = pickle.load(f)
################################################################################################################################################################################
# usage
build_vis_embed(embeddings_from_file, dictionary, "./vis_check")
# tensorboard --logdir=path/to/komy/check
# tensorboard --logdir=vis_check/check
