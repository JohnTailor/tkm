#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
import numpy as np
cimport numpy as np


cpdef _getTopicAssignments(np.ndarray[np.int32_t, mode="c"] docs, int ndocs, int nTopics, int winwid, np.ndarray[np.float32_t, ndim=2, mode="c"] p_w_t, np.ndarray[np.float32_t, ndim=2, mode="c"] nwt,np.ndarray[np.float32_t, ndim=2, mode="c"] p_d_t,np.ndarray[np.float32_t, ndim=2, mode="c"] new_p_d_t,np.float32_t alpha):
    _getTopicAssignmentsNoGIL(ndocs,&docs[0], nTopics, winwid,&p_w_t[0,0],&nwt[0,0],&p_d_t[0,0],&new_p_d_t[0,0],alpha)

cdef void _getTopicAssignmentsNoGIL(int ndocs,np.int32_t* d, int nTopics, int winwid, np.float32_t* p_w_t, np.float32_t* nwt,np.float32_t* p_t,np.float32_t* new_p_t, np.float32_t alpha) nogil:
        cdef int i, k, icl,doff,lend,istart,iend,iw
        cdef float csum,cmax
        cdef int pos =0
        cdef int dpoff=0
        cdef int woff=0
        cdef int woffs=0
        for i in xrange(ndocs): #go through all docs
           dpoff=i*nTopics
           lend = d[pos]
           istart=1+pos
           iend=pos+lend+1
           for doff in xrange(istart,iend): #go through one doc
                    woff=d[doff]*nTopics
                    cmax=0
                    for ind in xrange(max(istart,doff-winwid),min(iend,doff+winwid+1)): #walk through window of doc, find maximum score
                        woffs=d[ind]*nTopics
                        for k in xrange(nTopics):
                            if ((p_w_t[woffs+k]+p_w_t[woff+k]))* p_t[dpoff+k] > cmax: #Check if new max score?
                                icl=k
                                iw=woff
                                cmax=((p_w_t[woffs+k]+p_w_t[woff+k]))* p_t[dpoff+k]
                    nwt[woff+icl] += 1 #Assign word to topic
           #Compute doc-topic distribution
           for doff in xrange(1+pos,pos+lend+1):
                    woff=d[doff]*nTopics
                    for k in xrange(nTopics):  new_p_t[dpoff+k]+=p_w_t[woff+k]
           csum=0
           for k in xrange(nTopics):   csum+=new_p_t[dpoff+k]**alpha
           for k in xrange(nTopics):   new_p_t[dpoff+k] = (new_p_t[dpoff+k]**alpha) /csum
           pos+=d[pos]+1 #move to next doc