#Author: Johannes Schneider, University of Liechtenstein
#Paper: Topic modeling based on Keywords and Context, Please cite: https://arxiv.org/abs/1710.02650 (under submission as of 10/2017)
import numpy as np
import scipy
from scipy import stats
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy import random as ra
from stemming.porter import stem
from array import array

stemmMap={'ADJ':'a', 'ADJ_SAT':'a', 'ADV':'r', 'NOUN':'n', 'VERB':'v'}
stemmer = WordNetLemmatizer()

engStop = stopwords.words('english')
setStopWords = set(engStop)
import re

rep = {".": " ", "!": " ","?":" ",")":" ","(":" ",",":" ",";":" ",":":" ","`":" ","'":" ", '"':" "} # define desired replacements here
# use these three lines to do the replacement
rep = dict((re.escape(k), v) for k, v in rep.iteritems())
pattern = re.compile("|".join(rep.keys()))



def getUniqueTopics(p_w_t,thres):
    utop = [p_w_t[:, 0]]
    nTopics = p_w_t.shape[1]
    for i in range(1, nTopics):
        def myent(arr):  # compute KL distance, since non symmetric return addition
            return (scipy.stats.entropy(arr + 1e-8, np.transpose(p_w_t[:, i] + 1e-8)) + scipy.stats.entropy(
                np.transpose(p_w_t[:, i] + 1e-8), arr + 1e-8))
        tent = np.apply_along_axis(myent, 0, np.transpose(np.array(utop)))
        if np.min(tent) >= thres: utop.append(p_w_t[:, i])
    return utop


def computePerplexity(p_w_t,p_d_t, mdocs,getInitial=False,toprint=""):
        per =0
        for id,d in enumerate(mdocs):
           for i, wid in enumerate(d):
                pwidt = p_w_t[ wid,:]
                prob = np.fabs(np.dot(pwidt, p_d_t[id,:]))
                per += np.log(prob)
        normc = sum([len(d) for d in mdocs])
        per /= normc
        per = np.log(np.exp(-per))  # correct: np.exp(-perplexity/lenNew)
        return per


def tonp(modocs):
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

def wordsToIndex(docs): #use 1 for true
    mdocs = [array('I',[0]*len(docs[d])) for d in range(len(docs))]
    wCounts = {}
    for id, d in enumerate(docs):
        for iw, w in enumerate(d):
            wCounts[w]=wCounts.get(w,0)+1
    import operator
    sorted_co = sorted(wCounts.items(), key=operator.itemgetter(1),reverse=True) #create words in sorted manner by frequency, since freq are accesssed more, thus this reduces cache misses
    wordToiDict = {}
    iToWordDict = {}
    for i,(w,_) in enumerate(sorted_co):
        wordToiDict[w] = i
        iToWordDict[i] = w
    for id, d in enumerate(docs):
        for iw, w in enumerate(d):
            mdocs[id][iw] = wordToiDict[w]
    return mdocs,iToWordDict,wordToiDict


def processCorpus(corpus):
    mapWord = {}
    tokDocs = [tokenDoc(d, mapWord) for d in corpus]
    idocs, iToWordDict, wordToiDict = wordsToIndex(tokDocs)
    #Remove short docs and words that only occur once
    wco = {}
    mindoclen = 10
    for i,d in enumerate(idocs):
        if (len(d)<mindoclen): #filter short docs, they only cause trouble later on
            continue
        for w in d:
            if len(iToWordDict[w])>1: #remove words of length 1
                wco[w]=wco.get(w,0)+1

    wMoreOnce = set(range(len(iToWordDict))) - set([w for w,c in wco.iteritems() if c < 2]) #newMap = { w:i for i,w in enumerate(wMoreOnce)}
    newdocs = [[iToWordDict[w] for w in d if w in wMoreOnce]  for d in idocs] #remove unique words
    idocs, iToWordDict, wordToiDict = wordsToIndex(newdocs)
    return idocs,iToWordDict



def docToWords(doc):
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], doc).replace("  "," ").split(" ") #split into tokens
    matches = [t.lower() for t in text if len(t) > 1 and t.isalpha()] #lower case, remove non-letters, words of length 1
    matches = [t for t in matches if not t in setStopWords] #remove stop words
    matches = [stem(t) for t in matches] #stem words
    matches = [t for t in matches if not t in setStopWords]  # remove stop words again, due to stemming this seems to happen...
    return matches

def tokenDoc(doc, mapWord):
        words = []
        wo = docToWords(doc)
        for stWord in wo:
            if not stWord in mapWord:
                mapWord[stWord] = stWord
                words.append(stWord)
            else:
                words.append(mapWord[stWord])
        return words


def print_topics(p_w_t,iToWordDict):
    print "\n Printing up to 50 topics with up to 15 words each...\n---------------------------------------\n"
    ntopics=len(p_w_t[0])
    nw=len(p_w_t)
    for t in range(min(ntopics,50)):
        lp = [(p_w_t[wid][t], wid) for wid in range(nw)]
        lp.sort(reverse=True)
        tstr = str(t) + ":    "
        for (p, wid) in lp[0:15]:
            if p > 0.001:
                tstr += '%s %.3f, ' % (iToWordDict[wid], p*10)
            else:
                tstr += '%s %.5f, ' % (iToWordDict[wid], p*10)
        print tstr
