#Author: Johannes Schneider, University of Liechtenstein
#Paper: Topic modeling based on Keywords and Context, Please cite: https://arxiv.org/abs/1710.02650 (under submission as of 10/2017)
import random
import time

import _topicAssign
import numpy as np
import scipy.stats

import algTools

class TKMCore:
    def computeKeywordScore(self, nwt, human=False):
        nw = np.sum(nwt, axis=1)  # number of times word occurs overall
        entropy_p_t_w = np.apply_along_axis(scipy.stats.entropy, 1, nwt + 1e-30)  # entropy of topic-word distr, small constant if word has 0 occurrences in all topics
        maxEntropy = np.log2(np.clip(nw, 2, self.nTopics))  # For concentration normalization: min (log #topics, log #times word occurs)
        concentration = (maxEntropy / (1 + entropy_p_t_w)) ** (1.5)
        if human: new_f_w_t = nwt * (concentration.reshape(concentration.shape[0], 1)) #human readable: stronger focus on frequncy
        else:   new_f_w_t = np.log2(1 + nwt + self.beta) * (concentration.reshape(concentration.shape[0], 1)) #for inference/extracting keywords
        normconst = 1e-30 + np.sum(new_f_w_t, axis=0)  # sum across words for a topic # sum_z p(w|d,z)*p(z|d)
        for t in range(normconst.shape[0]): new_f_w_t[:, t] /= normconst[t]
        return new_f_w_t


    #perform one E-Step and one M-Step
    def oneStep(self, pdt):
        new_p_d_t = np.zeros((self.ndocs,self.nTopics),dtype=np.float32)  #New prob distributions
        nwt = np.zeros((self.nw, self.nTopics), dtype=np.float32)
        _topicAssign._getTopicAssignments(self.npdocs, self.ndocs, self.nTopics, self.winwid, self.f_w_t, nwt, pdt, new_p_d_t, self.alpha)
        new_f_w_t=self.computeKeywordScore(nwt)
        return new_f_w_t,new_p_d_t,nwt


    #mdocs are a list of list of numbers
    def run(self, mdocs,nWords, nTopics,winwid, alpha,beta, convconst=0.05, miter=500, mseed=int(time.time() % 10000),klDistThres=0.25):
        self.npdocs= algTools.tonp(mdocs)
        self.ndocs = len(mdocs)
        self.nTopics = nTopics
        self.nw = nWords
        self.beta = beta
        self.alpha = alpha
        self.winwid = winwid
        np.random.seed(mseed)
        random.seed(np.random.random())

        #initialize
        self.f_w_t = np.ones((self.nw, self.nTopics), dtype=np.float32)
        self.p_d_t = np.full((self.ndocs, self.nTopics), 1.0 / self.nTopics, dtype=np.float32)
        self.f_w_t += np.random.random((self.nw, self.nTopics)) / 75  #~ 1per cent randomness
        normw = np.sum(self.f_w_t, axis=0)
        for t in range(self.nTopics): self.f_w_t[:, t] = self.f_w_t[:, t] / normw[t]

        #iterate until convergence
        lastper=1e20
        for i in range(1,miter+1):
                self.globiter=i
                (self.f_w_t, self.p_d_t, _) = self.oneStep(self.p_d_t)
                if i%7 ==0: #check only every couple of iterations to get more stable results
                        per = algTools.computePerplexity(self.f_w_t, self.p_d_t, mdocs, None)
                        if ((lastper-per)/lastper) < convconst: break
                        lastper=per
        utop = algTools.getUniqueTopics(self.f_w_t, klDistThres)
        self.nTopics = len(utop)
        self.f_w_t = np.transpose(np.array(utop)).copy(order='C')
        self.p_d_t,self.nwtTrainLastIteration = self.getTopics(mdocs)

    #get Topics of new,unseen  documents without adjusting existing distribution
    def getTopics(self,mdocs):
        self.npdocs = algTools.tonp(mdocs)
        self.ndocs = len(mdocs)
        p_d_t = np.full((self.ndocs, self.nTopics), 1.0 / self.nTopics, dtype=np.float32)
        (_, p_d_t, nwtTrain) = self.oneStep(p_d_t)
        return p_d_t,nwtTrain

    #Training distributions
    def get_p_d_t(self): return self.p_d_t

    def get_f_w_t(self): return self.f_w_t

    def get_f_w_t_hu(self):
        return self.computeKeywordScore(self.nwtTrainLastIteration, True)





