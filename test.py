# Author: Johannes Schneider, University of Liechtenstein
# Paper: Topic modeling based on Keywords and Context
# Please cite: https://arxiv.org/abs/1710.02650 (accepted at SDM 2018)

import re

import nltk
from nltk.corpus import brown

import TKMCore
import algTools


def readBrownDataset():
    nltk.download("brown")
    documents = brown.fileids()
    docs = []
    for doc in documents:
        if len(brown.categories(doc)) == 1:
            d = brown.raw(doc).replace("\n", " ")
            d = re.sub(r"/[A-Za-z0-9_-]+ ", " ", d)  #The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn") #.replace("/at","").replace("/nn-tl","").replace("/nn-hp","").replace("/np-hl","").replace("/nn","").replace("/vbd","").replace("/in","").replace("/jj","").replace("/hvz","").replace("/cs","").replace("/nps","").replace("/nr","").replace("/np-tl","").replace("/md","").replace("/np","").replace("/cd-hl","").replace("/vbn","").replace("/np-tl","").replace("/dti","").replace("--/--","")
            docs.append(d)
    return docs


print("\nDownloading test data set using Python's NLTK library...")
docs = readBrownDataset()
print("\nPreprocessing data set...")
# Transform data set with words into sequence of numbers
m_docs, id2word = algTools.process_corpus(docs)

print([id2word[_id] for _id in m_docs[0][:50]])
print(["{}: {}".format(_id, word) for _id, word in id2word.items()][:10])

print("\nRunning TKM... - Takes 1 - 2 minutes")
tkmc = TKMCore.TKMCore(
    m_docs=m_docs,
    n_words=len(id2word),
    n_topics=20,
    winwid=7,
    alpha=7,
    beta=0.08
)
tkmc.run(convergence_constant=0.08, mseed=4848)
print("\nPrinting Topics with Human Weights...")
algTools.print_topics(p_w_t=tkmc.get_f_w_t_hu(), id2word=id2word)

