# from snownlp import SnowNLP
#
#
# text = u'这个东西真心很赞'
# s = SnowNLP(text)
# # 正负面
# score = s.sentiments
# # 提取关键字
# kw = s.keywords(3)
# # 提取摘要
# su = s.summary(3)
# print(score)
# print(kw)
# print(su)
import jieba
import numpy as np

# 打开词典文件，返回列表
def open_dict(Dict='hahah',path = r''):
    path = path + '%s.txt' %Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def judgeodd(num):
    if num % 2 == 0:
        return 'even'
    else:
        return 'odd'

deny_word = open_dict(Dict='否定词')
posdict = open_dict(Dict='positive')
negdict = open_dict(Dict = 'negative')

degree_word = open_dict(Dict = '程度级别词语',path=r'')
mostdict = degree_word[degree_word.index('extreme')+1: degree_word.index('very')] #权重4，即在情感前乘以3
verydict = degree_word[degree_word.index('very')+1: degree_word.index('more')] #权重3
moredict = degree_word[degree_word.index('more')+1: degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1: degree_word.index('last')]#权重0.5

def sentiment_score_list(dataset):
    seg_sentence = dataset.split('。')

    count1 = []
    count2 = []
    for sen in seg_sentence: # 循环遍历每一个评论
        segtmp = jieba.lcut(sen, cut_all=False) # 把句子进行分词，以列表的形式返回
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        poscount = 0 # 积极词的第一次分值
        poscount2 = 0 # 积极反转后的分值
        poscount3 = 0 # 积极词的最后分值（包括叹号的分值）
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        for word in segtmp:
            if word in posdict: # 判断词语是否是情感词
                poscount +=1
                c = 0
                for w in segtmp[a:i]: # 扫描情感词前的程度词
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                       poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word: c+= 1
                if judgeodd(c) == 'odd': # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i+1
            elif word in negdict: # 消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1
            elif word == '！' or word == '!': # 判断句子是否有感叹号
                for w2 in segtmp[::-1]: # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict or negdict:
                        poscount3 += 2
                        negcount3 += 2
                        break
            i += 1

            # 以下是防止出现负数的情况
            pos_count = 0
            neg_count = 0
            if poscount3 <0 and negcount3 > 0:
                neg_count += negcount3 - poscount3
                pos_count = 0
            elif negcount3 <0 and poscount3 > 0:
                pos_count = poscount3 - negcount3
                neg_count = 0
            elif poscount3 <0 and negcount3 < 0:
                neg_count = -pos_count
                pos_count = -neg_count
            else:
                pos_count = poscount3
                neg_count = negcount3
            count1.append([pos_count,neg_count])
        count2.append(count1)
        count1=[]

    return count2

def sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        score_array =  np.array(review)
        Pos = np.sum(score_array[:,0])
        Neg = np.sum(score_array[:,1])
        AvgPos = np.mean(score_array[:,0])
        AvgPos = float('%.lf' % AvgPos)
        AvgNeg = np.mean(score_array[:, 1])
        AvgNeg = float('%.1f' % AvgNeg)
        StdPos = np.std(score_array[:, 0])
        StdPos = float('%.1f' % StdPos)
        StdNeg = np.std(score_array[:, 1])
        StdNeg = float('%.1f' % StdNeg)
        score.append([Pos,Neg,AvgPos,AvgNeg,StdPos,StdNeg])
    return score

data = '用了几天又来评价的，手机一点也不卡，玩荣耀的什么的不是问题，充电快，电池够大，玩游戏可以玩几个小时，待机应该可以两三天吧，很赞'
data2 = '不知道怎么讲，真心不怎么喜欢，通话时声音小，新手机来电话竟然卡住了接不了，原本打算退，刚刚手机摔了，又退不了，感觉不会再爱，像素不知道是我不懂还是怎么滴 感觉还没z11mini好，哎要我怎么评价 要我如何喜欢努比亚 太失望了'

print(sentiment_score(sentiment_score_list(data)))
# print(sentiment_score(sentiment_score_list(data2))