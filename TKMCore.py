# Author: Johannes Schneider, University of Liechtenstein
# Paper: Topic modeling based on Keywords and Context
# Please cite: https://arxiv.org/abs/1710.02650 (accepted at SDM 2018)

import random
import time

import numpy as np
from scipy.stats import entropy

import _topicAssign
import algTools


class TKMCore:

    def __init__(self, m_docs, n_words, n_topics, winwid, alpha, beta):
        """
        :param m_docs: Document corpus transformed so that integer IDs
                       represent each word in the vocabulary.
                       Type: List[List[int]]
        :param n_words: Number of unique words in the corpus' vocabulary
        :param n_topics: Maximum number of topics to be assigned to the corpus
        :param winwid: ???
        :param alpha: Topic concentration prior: the larger α, the more
                      concentrated the topics - i.e. fewer topics per document
        :param beta: Prior with states that each word is assumed to occur at
                     least β times in each topic
        """
        self.m_docs = m_docs
        self.np_docs = algTools.to_np(m_docs)
        self.n_docs = len(m_docs)
        self.n_topics = n_topics
        self.n_words = n_words
        self.beta = beta
        self.alpha = alpha
        self.winwid = winwid

        self._initialise()

    def _initialise(self):
        """
        Initialise the keyword score and document topic probability matrices
        """
        self.f_w_t = np.ones((self.n_words, self.n_topics), dtype=np.float32)
        self.p_d_t = np.full((self.n_docs, self.n_topics), 1.0 / self.n_topics,
                             dtype=np.float32)
        self.f_w_t += np.random.random((self.n_words, self.n_topics)) / 75  # ~1% randomness
        normw = np.sum(self.f_w_t, axis=0)
        for t in range(self.n_topics):
            self.f_w_t[:, t] = self.f_w_t[:, t] / normw[t]

    def compute_keyword_score(self, nwt, human=False):
        """
        :param nwt: Vector representing a word's assignment count to the
                    different topics
        :param human: Set to True for a more human readable interpretation by
                      setting a stronger focus on frequency
        :return: The updated keyword score matrix
        """
        nw = np.sum(nwt, axis=1)  # Number of times word occurs overall
        # Entropy of topic-word distribution.
        # Is a small constant if word has 0 occurrences in all topics.
        entropy_p_t_w = np.apply_along_axis(entropy, 1, nwt + 1e-30)
        # For concentration normalization:
        #   min (log #topics, log #times word occurs)
        max_entropy = np.log2(np.clip(nw, 2, self.n_topics))
        concentration = (max_entropy / (1 + entropy_p_t_w)) ** 1.5

        if human:  # Human readable: stronger focus on frequency
            new_f_w_t = nwt * (concentration.reshape(concentration.shape[0], 1))
        else:  # For inference/extracting keywords
            new_f_w_t = np.log2(1 + nwt + self.beta) * \
                        (concentration.reshape(concentration.shape[0], 1))

        # Sum across words for a topic # sum_z p(w|d,z)*p(z|d)
        norm_const = 1e-30 + np.sum(new_f_w_t, axis=0)
        for t in range(norm_const.shape[0]):
            new_f_w_t[:, t] /= norm_const[t]

        return new_f_w_t

    def one_step(self, pdt):
        """Performs one E-step and one M-step"""
        # Calculate new probability distribution
        new_p_d_t = np.zeros((self.n_docs, self.n_topics), dtype=np.float32)
        nwt = np.zeros((self.n_words, self.n_topics), dtype=np.float32)
        _topicAssign._getTopicAssignments(
            self.np_docs, self.n_docs, self.n_topics, self.winwid, self.f_w_t,
            nwt, pdt, new_p_d_t, self.alpha)
        new_f_w_t = self.compute_keyword_score(nwt)

        return new_f_w_t, new_p_d_t, nwt

    def run(self, convergence_constant=0.05, max_iter=500, kl_threshold=0.25,
            mseed=int(time.time() % 10000)):
        """
        Run the TKM algorithm until convergence, and update the number of
        topics, topic keyword score matrix and the document-topic probability
        distribution.

        :param convergence_constant: Convergence constant
        :param max_iter: Maximum number of iterations
        :param kl_threshold: Maximum Kullback-Leibler divergence
        :param mseed: Random seed
        :return: None
        """
        np.random.seed(mseed)
        random.seed(np.random.random())

        # Iterate until convergence
        prev_perplexity = 1e20
        for i in range(1, max_iter + 1):
            self.f_w_t, self.p_d_t, _ = self.one_step(self.p_d_t)
            # Only check for convergence every few iterations for increased
            # stability
            if i % 7 == 0:
                perplexity = algTools.compute_perplexity(self.f_w_t,
                                                         self.p_d_t,
                                                         self.m_docs)
                if (prev_perplexity - perplexity) / prev_perplexity < convergence_constant:
                    print("Number of iters: %s" % i)
                    break
                prev_perplexity = perplexity

        unique_topics = algTools.get_unique_topics(self.f_w_t, kl_threshold)
        self.n_topics = len(unique_topics)
        self.f_w_t = np.transpose(np.array(unique_topics)).copy(order='C')
        self.p_d_t, self.nwt_latest_iteration = self.get_topics(self.m_docs)

    def get_topics(self, m_docs):
        """
        Get the topics of new, unseen documents without adjusting the current
        distribution
        :param m_docs: Unseen documents transformed so that words are replaced
                       by integer representations using the same dictionary
                       (ID-to-word mapping) as used when training
        :return: Updated matrices for document topic probabilities and number
                 of assignments to each topic for each word
        """
        self.np_docs = algTools.to_np(m_docs)
        self.n_docs = len(m_docs)
        p_d_t = np.full((self.n_docs, self.n_topics), 1.0 / self.n_topics,
                        dtype=np.float32)
        _, p_d_t, nwt_updated = self.one_step(p_d_t)
        return p_d_t, nwt_updated

    # Training distributions
    def get_p_d_t(self): return self.p_d_t

    def get_f_w_t(self): return self.f_w_t

    def get_f_w_t_hu(self):
        return self.compute_keyword_score(self.nwt_latest_iteration, True)





