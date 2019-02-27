# Author: Johannes Schneider, University of Liechtenstein
# Paper: Topic modeling based on Keywords and Context
# Please cite: https://arxiv.org/abs/1710.02650 (accepted at SDM 2018)

import operator
import re
from array import array
from collections import defaultdict

import numpy as np

import scipy
from scipy import stats

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

#from stemming.porter import stem

stemmMap = {'ADJ': 'a', 'ADJ_SAT': 'a', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

stopwords_eng = stopwords.words('english')
stopwords_set = set(stopwords_eng)

# Define replacements and compile regex pattern
replacements = {
    ".": " ",
    "!": " ",
    "?": " ",
    ")": " ",
    "(": " ",
    ",": " ",
    ";": " ",
    ":": " ",
    "`": " ",
    "'": " ",
    "\"": " "
}
replacements = dict((re.escape(k), v) for k, v in replacements.items())
pattern = re.compile("|".join(replacements.keys()))


def get_unique_topics(p_w_t, kl_threshold):
    """
    Get the remaining unique topics, after filtering out those that are not
    significantly different, as set by the Kulback-Leibler distance threshold

    :param p_w_t: Word-topic probabilities matrix
    :param kl_threshold: Kulback-Leibler distance threshold for significantly
                         differing topics
    :returns: Filtered list of topics
    """
    print("p_w_t:")
    print(p_w_t[:5])
    unique_topics = [p_w_t[:, 0]]
    n_topics = p_w_t.shape[1]
    for i in range(1, n_topics):
        def kl_distance(topics_arr):
            return (scipy.stats.entropy(topics_arr + 1e-8,
                                        np.transpose(p_w_t[:, i] + 1e-8))
                    + scipy.stats.entropy(np.transpose(p_w_t[:, i] + 1e-8),
                                          topics_arr + 1e-8))
        topic_entropy = np.apply_along_axis(kl_distance, 0,
                                            np.transpose(np.array(unique_topics)))
        if np.min(topic_entropy) >= kl_threshold:
            unique_topics.append(p_w_t[:, i])
    return unique_topics


def compute_perplexity(p_w_t, p_d_t, m_docs):
    perplexity = 0
    for _id, doc in enumerate(m_docs):
        for i, word_id in enumerate(doc):
            pwidt = p_w_t[word_id, :]
            p = np.fabs(np.dot(pwidt, p_d_t[_id, :]))
            perplexity += np.log(p)
    normalisation_const = sum([len(d) for d in m_docs])
    perplexity /= normalisation_const
    perplexity = np.log(np.exp(-perplexity))  # correct: np.exp(-perplexity/lenNew)
    return perplexity


def to_np(modocs):
    mdocs = [np.frombuffer(d, dtype=np.int32) for d in modocs]
    toLen = sum([len(d) for d in mdocs])
    npdocs = np.zeros(toLen + len(mdocs), dtype=np.intc)
    i = 0
    for d in mdocs:
        npdocs[i] = len(d)
        i += 1
        for w in d:
            npdocs[i] = w
            i += 1
    return npdocs


def words_to_index(docs): #use 1 for true
    mdocs = [array('I', [0]*len(doc)) for doc in docs]
    word_counts = defaultdict(int)
    for doc_id, doc in enumerate(docs):
        for word_i, word in enumerate(doc):
            word_counts[word] += 1

    # Sort words by frequency, since this is accessed more. This setup reduces
    # cache misses going forwards.
    sorted_counts = sorted(word_counts.items(), key=operator.itemgetter(1),
                           reverse=True)
    word2id = {}
    id2word = {}
    for i, (word, _) in enumerate(sorted_counts):
        word2id[word] = i
        id2word[i] = word
    for doc_id, doc in enumerate(docs):
        for word_i, word in enumerate(doc):
            mdocs[doc_id][word_i] = word2id[word]

    return mdocs, id2word, word2id


def process_corpus(corpus, min_doc_length=10):
    """
    Clean corpus, and then transform words into integer representations.
    Cleaning entails removing stop words, normalising and stemming words, and
    filtering out documents shorter than specified length, as well as filtering
    out unique words.

    :param corpus: List strings, where each string is document of words
    :param min_doc_length: Minimum number of tokens a document needs to contain

    :return: Tuple of the transformed corpus and an id2word dictionary
    """
    word_map = {}
    tokenized_corpus = [tokenize_and_update_mapping(doc, word_map)
                        for doc in corpus]
    transformed_corpus, id2word, word2id = words_to_index(tokenized_corpus)

    # Remove short docs and words that only occur once, and update word counts
    word_counts = defaultdict(int)
    for i, doc in enumerate(transformed_corpus):
        # Filter out short docs, they only cause trouble later on
        if len(doc) < min_doc_length:
            continue
        for word_id in doc:
            if len(id2word[word_id]) > 1:  # Filter out words of length 1
                word_counts[word_id] += 1

    # Gather list of IDs for words that occur more than once
    non_unique_word_ids = set(range(len(id2word))) - \
                          set([w for w, c in word_counts.items() if c < 2])

    # Filter out unique words
    filtered_docs = [[id2word[word_id] for word_id in doc if word_id in non_unique_word_ids]
                     for doc in transformed_corpus]

    # Return the final corpus with words transformed into their integer
    # representations, as well as the id2word dictionary
    transformed_corpus, id2word, word2id = words_to_index(filtered_docs)
    return transformed_corpus, id2word


def tokenize(doc):
    tokens = pattern.sub(lambda m: replacements[re.escape(m.group(0))], doc)\
        .replace("  ", " ")\
        .split(" ")
    # lower case, remove non-letters and words of length 1
    matches = [t.lower() for t in tokens if len(t) > 1 and t.isalpha()]
    # remove stop words
    matches = [t for t in matches if t not in stopwords_set]
    # stem words
    matches = [stemmer.stem(t) for t in matches]
    # remove stop words again, due to stemming this seems to happen...
    matches = [t for t in matches if t not in stopwords_set]
    return matches


def tokenize_and_update_mapping(doc, word_map):
    words = []
    tokens = tokenize(doc)
    for word in tokens:
        if word not in word_map:
            word_map[word] = word
            words.append(word)
        else:
            words.append(word_map[word])
    return words


def print_topics(p_w_t, id2word, max_topics=50, max_words=15):
    """
    Print the most probable words for each topic, as limited by `max_word` and
    `max_topics`
    :param p_w_t: Word-topic probability matrix
    :param id2word: Dictionary mapping IDs to words
    :param max_topics: Maximum number of topics to print
    :param max_words: Maximum number of words to print per topic
    """
    display_str = "Printing up to {} topics with up to {} words each..."\
        .format(max_topics, max_words)
    print("\n{}\n{}\n".format(display_str, "-" * len(display_str)))

    n_topics = len(p_w_t[0])
    num_words = len(p_w_t)
    for topic_i in range(min(n_topics, max_topics)):
        word_topic_probabilities = [(p_w_t[word_id][topic_i], word_id)
                                    for word_id in range(num_words)]
        word_topic_probabilities.sort(reverse=True)

        topic_str = str(topic_i) + ":\t"
        for p, word_id in word_topic_probabilities[0:max_words]:
            if p > 0.001:
                topic_str += '%s %.3f, ' % (id2word[word_id], p * 10)
            else:
                topic_str += '%s %.5f, ' % (id2word[word_id], p * 10)
        print(topic_str)
