#!/usr/bin/env python
# -*- coding: utf-8  -*-
#从词向量模型中提取文本特征向量
'''
获取特征词向量的主要步骤如下：

+ 1）读取模型词向量矩阵；
+ 2）遍历语句中的每个词，从模型词向量矩阵中抽取当前词的数值向量，一条语句即可得到一个二维矩阵，行数为词的个数，列数为模型设定的维度；
+ 3）根据得到的矩阵计算**矩阵均值**作为当前语句的特征词向量；
+ 4）全部语句计算完成后，拼接语句类别代表的值，写入csv文件中。
'''
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim

# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
    

# 构建文档词向量 
def buildVecs(filename,model):
    fileVecs = []
    with codecs.open(filename, 'r', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
    return fileVecs   

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    fdir = r'C:\Users\17801\PycharmProjects\NLPTT\Word2VecNlp\word2vec-Chinese-master'
    inp = fdir + '\zhwiki.word2vec.vectors'     #word2vec词向量文件,通过维基中文百科35w条数据训练的得出
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    
    posInput = buildVecs('2000_pos_cut.txt',model)
    negInput = buildVecs('2000_neg_cut.txt',model)

    # use 1 for positive sentiment， 0 for negative
    Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))

    X = posInput[:]
    for neg in negInput:
        X.append(neg)
    X = np.array(X)

    # write in file   
    df_x = pd.DataFrame(X)
    df_y = pd.DataFrame(Y)
    data = pd.concat([df_y,df_x],axis = 1)
    #print data
    data.to_csv('2000_data.csv')
    
#代码执行完成后，得到一个名为2000_data.csv的文件，第一列为类别对应的数值（1-pos, 0-neg），第二列开始为数值向量，每一行代表一条评论。
    


