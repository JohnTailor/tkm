Topic Modeling based on Keywords and Context
--------------------------------------------

The Topic Keyword Model is a novel topic model that works quite differently from LDA from Blei et al. and other topic models.

Please check and cite our paper:
Johannes Schneider, Michail Vlachos. Topic Modeling based on Keywords and Context, SIAM International Conference on Data Mining (SDM), 2018

PDF: https://arxiv.org/abs/1710.02650


Usage:
------
Install Cython

Run "python setup.py build_ext --inplace" in the /TKM folder to compile Cython code, ie. "_topicAssign.c" and "_topicAssign.pyd" files will be created

Run "python test.py"


Related projects:
----------------
BTM model https://github.com/xiaohuiyan/BTM
