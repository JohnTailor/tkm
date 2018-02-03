#Author: Johannes Schneider, University of Liechtenstein
#Paper: Topic modeling based on Keywords and Context, Please cite: https://arxiv.org/abs/1710.02650 (under submission as of 10/2017)


import TKMCore
import algTools

def readBrownDataset():
    import nltk
    nltk.download("brown")
    from nltk.corpus import brown
    documents = brown.fileids()
    docs=[]
    import re
    for fi in documents:
            if len(brown.categories(fi)) ==1:
                d= brown.raw(fi).replace("\n"," ")
                d=re.sub(r"/[A-Za-z0-9_-]+ "," ",d)#	The/at Fulton/np-tl County/nn-tl Grand/jj-tl Jury/nn-tl said/vbd Friday/nr an/at investigation/nn") #.replace("/at","").replace("/nn-tl","").replace("/nn-hp","").replace("/np-hl","").replace("/nn","").replace("/vbd","").replace("/in","").replace("/jj","").replace("/hvz","").replace("/cs","").replace("/nps","").replace("/nr","").replace("/np-tl","").replace("/md","").replace("/np","").replace("/cd-hl","").replace("/vbn","").replace("/np-tl","").replace("/dti","").replace("--/--","")
                docs.append(d)
    return docs


print "\nDownloading Testdataset using Python's NLTK library..."
docs=readBrownDataset()
print "\nPreprocessing Dataset..."
idocs,iToWord=algTools.processCorpus(docs) #Turn dataset with words into sequence of numbers
print "\nRunning TKM... - Takes 1 - 2 minutes"
tkmc=TKMCore.TKMCore()
#( mdocs,nWords, nTopics,winwid, alpha,beta, convconst=0.05, miter=500, mseed=int(time.time() % 10000),klDistThres=0.25):
tkmc.run(idocs,len(iToWord),20,7,8,0.08)
print "\nPrinting Topics with Human Weights..."
algTools.print_topics(tkmc.get_f_w_t_hu(),iToWord)

