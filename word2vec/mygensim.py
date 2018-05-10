# import modules and set up logging
import logging

from gensim.models import KeyedVectors
from gensim.models import word2vec


# in jupiter ------------------------------------------------------------
# !pip install gensim
# !curl http://mattmahoney.net/dc/text8.zip > text8.zip
# !unzip text8.zip
# !curl https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip > source-archive.zip
# !unzip -p source-archive.zip  word2vec/trunk/questions-words.txt > questions-words.txt
# !rm text8.zip source-archive.zip
# in jupiter ------------------------------------------------------------

def train():
    # load up unzipped corpus from http://mattmahoney.net/dc/text8.zip
    sentences = word2vec.Text8Corpus('D:\preferences\jetBrains\pycharm\\udacity\\text8.data')
    # train the skip-gram model; default window=5
    model = word2vec.Word2Vec(sentences, size=200)
    # ... and some hours later... just as advertised...
    model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

    # pickle the entire model to disk, so we can load&resume training later
    model.save('./gensim_tmp/text8.model')
    # store the learned weights, in a format the original C tool understands
    model.wv.save_word2vec_format('./gensim_tmp/text8.model.bin', binary=True)
    # or, import word weights created by the (faster) C word2vec
    # this way, you can switch between the C/Python toolkits easily


def eval():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = KeyedVectors.load_word2vec_format('./gensim_tmp/text8.model.bin', binary=True)

    # "boy" is to "father" as "girl" is to ...?
    model.most_similar(['girl', 'father'], ['boy'], topn=3)

    more_examples = ["he his she", "big bigger bad", "going went being"]
    for example in more_examples:
        a, b, x = example.split()
        predicted = model.most_similar([x, b], [a])[0][0]
        print("'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted))

    # which word doesn't go with the others?
    model.doesnt_match("breakfast cereal dinner lunch".split())

    print(model.accuracy('./eval_data/questions-words.txt'))  # 0.3250989843346531 :(

    results = [(149, 506), (272, 1452), (33, 268), (165, 1571), (237, 306), (100, 756), (58, 306), (744, 1260), (184, 506), (286, 992), (743, 1371), (367, 1332), (439, 992)]

    sumN, sumD = 0, 0
    for n, d in results:
        sumN += n
        sumD += d

    print("{}/{} {}".format(sumN, sumD, sumN / sumD))


eval_path = './eval_data/questions-words.txt'


def read_analogies(dictionary):
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
                words = [w.strip() for w in words]

                if any([word not in dictionary for word in words]) or len(words) != 4:
                    questions_skipped += 1
                else:
                    questions.append(words)

                count_per_class += 1

    question_classes[-1] = (question_classes[-1][0], count_per_class)
    count_total += count_per_class

    print("Eval analogy file: ", eval_path)
    print("Questions: ", len(questions))
    assert len(questions) + questions_skipped == count_total
    print("Skipped: ", questions_skipped)
    print(question_classes)
    return questions


def my_gs_eval():
    model = KeyedVectors.load_word2vec_format('./gensim_tmp/text8.model.bin', binary=True)
    analogy_questions = read_analogies(model.vocab)
    count = 0
    predictions = []
    for analogy in analogy_questions:
        a, b, c, label = analogy
        predicted = model.most_similar([c, b], [a], topn=1)

        if label in list([pred[0] for pred in predicted]):
            count += 1

        predictions.append(predicted)
    print(count / len(analogy_questions), "{}/{}".format(count, len(analogy_questions)))

    return predictions


def getModelObjectsAsNumpy():
    model = KeyedVectors.load_word2vec_format('./gensim_tmp/text8.model.bin', binary=True)

    word2in = dict()
    for key, value in model.vocab.items():
        word2in[key] = value.index

    # "boy" is to "father" as "girl" is to ...?

    import pickle
    import gzip

    f = gzip.open("./gensim_tmp/gensim.word2vec.embedd.pkl.gz", "w")
    pickle.dump((word2in, model.syn0), f)
    f.close()

# train()
# eval()
# run_my_eval()
# getModelObjectsAsNumpy()
