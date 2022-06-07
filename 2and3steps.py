#由于Python2和3部分函数不兼容，特此重写一个新的文件，本文件将实现原2、3部分的分词和停用词处理的功能
import jieba
import re
import time
from collections import Counter
import csv
import pandas as pd
def Chinese(text):      #处理函数
    cleaned = re.findall(r'[\u4e00-\u9fa5]+', text)  #返回列表,用正则表达式筛选出中文
    cleaned = ''.join(cleaned)                      #拼接成字符串
    return cleaned
# 创建停用词list
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r',encoding='utf-8').readlines()]
    return stopwords
stopwords = stopwordslist('data\\stopWord.txt')  # 这里加载停用词的路径
def cutSentence(originalPath,cutFilePath):      #实现分词和停用词筛

    all_words = ""
    f = open(cutFilePath, 'w', encoding='utf-8')
    for line in open(originalPath, encoding='utf-8'):
        cut_words = ""
        line.strip('\n')
        line = Chinese(line)            #调用清洗函数
        seg_list = jieba.cut(line,cut_all=False)
        for word in seg_list:
            if word not in stopwords:
                if word != '\t':
                    cut_words += ("".join(word))
                    cut_words += " "
        f.write(cut_words)
        f.write('\n')
        all_words += cut_words
    f.close()
    # 输出结果
    all_words = all_words.split()
    #print(all_words)
    print("文件处理成功")
if __name__ == '__main__':
    sourceFile = '2000_neg.txt'
    targetFile = '2000_neg_cut.txt'
    positiveFile = '2000_pos.txt'
    cutPosFile = '2000_pos_cut.txt'
    cutSentence(sourceFile,targetFile)
    cutSentence(positiveFile, cutPosFile)
