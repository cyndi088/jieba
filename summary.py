# -*- coding: utf-8 -*-
import time
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator

f = open('stopword.txt', encoding='utf-8')  # 停止词
stopwords = f.readlines()
stopwords = [i.replace("\n", "") for i in stopwords]


def cleanData(name):
    setlast = jieba.cut(name, cut_all=False)
    seg_list = [i.lower() for i in setlast if i not in stopwords]
    return " ".join(seg_list)


def calculateSimilarity(sentence, doc):  # 根据句子和句子，句子和文档的余弦相似度
    if doc == []:
        return 0
    vocab = {}
    for word in sentence.split():
        vocab[word] = 0  # 生成所在句子的单词字典，值为0

    docInOneSentence = ''
    for t in doc:
        docInOneSentence += (t + ' ')  # 所有剩余句子合并
        for word in t.split():
            vocab[word] = 0  # 所有剩余句子的单词字典，值为0

    cv = CountVectorizer(vocabulary=vocab.keys())

    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])
    return cosine_similarity(docVector, sentenceVector)[0][0]


# data = open("4.txt", encoding='utf-8')  # 测试文件
# texts = data.readlines()  # 读行
# texts = [i[:-1] if i[-1] == '\n' else i for i in texts]

data = '近期，曾创下日订单量过万纪录的烘焙O2O品牌香送面包疑因资金链断裂停止运营，留下的是数百名用户尚待退还的储值金、拖欠近1个月的30万元房租、180万元的员工工资和各种外债。 　　2015年，被称为“面包界21Cake”的香送面包在北京成立，试图凭借中央厨房+物流配送的O2O模式省去高昂的门店租金，进而提升产品品质和客户体验。辉煌时，香送曾拥有14个配送站点、50万用户规模、90%的月订单增长率，并计划引入资本将业务延伸至其他城市。 　　然而运行近3年后，香送却难敌研发、地推、配送等高成本带来的压力，成为又一个死在O2O幻想中的创业项目。 　　总部办公室人去楼空 　　“我们将无法再继续为大家服务了。”5月25日晚，一条香送面包停运的消息在香送用户的朋友圈中流传，其中写道：“经历了两年多的运营，我们遇到了很多坎……这一次我们确实撑不过去了”，“关于停运带来的后续事宜，会有专门团队来处理。” 　　“5月26日有人私信我，我才知道香送倒闭了。”一位香送会员告诉新京报记者，目前其储值卡内尚有425元余额未退还，办卡时赠送的2张蛋糕券也用不了了。5月29日，她按照香送微信公号给的退款邮箱发去邮件，至今尚未有人联系处理。 　　在香送原本冷清的官方微博下，要求退钱的消费者也自5月26日起密集出现。有人留言称香送“不发声明，不退钱，还接单，任何客服联系不上”，还有人调侃“倒闭了连个声明都没有，幸好是给熟人的蛋糕，不然尴尬死”。 　　曾在香送某片区负责地推工作的陈伟(化名)向新京报记者透露，目前其建立的香送用户“维权群”中已有300多人，待退款金额共计五六万元。而另一个香送用户自发组织的“维权群”中已有400余人。 　　工商信息显示，北京香送电子商务有限公司已在今年5月31日被北京朝阳工商分局列入“经营异常名录”，原因是“通过登记的住所或经营场所无法取得联系”。'
texts = data.split(' ')
print(texts)
print(len(texts))

sentences = []
clean = []
originalSentenceOf = {}

start = time.time()

# Data cleansing
for line in texts:
    parts = line.split('。')[:-1]  # 句子拆分
    # print(parts)
    for part in parts:
        cl = cleanData(part)  # 句子切分以及去掉停止词
        # print(cl)
        sentences.append(part)  # 原本的句子
        clean.append(cl)  # 干净有重复的句子
        originalSentenceOf[cl] = part  # 字典格式
setClean = set(clean)  # 干净无重复的句子

# calculate Similarity score each sentence with whole documents
scores = {}
for data in clean:
    temp_doc = setClean - set([data])  # 在除了当前句子的剩余所有句子
    score = calculateSimilarity(data, list(temp_doc))  # 计算当前句子与剩余所有句子的相似度
    scores[data] = score  # 得到相似度的列表
    # print(score)


# calculate MMR
n = 25 * len(sentences) / 100  # 摘要的比例大小
alpha = 0.7
summarySet = []
while n > 0:
    mmr = {}
    # kurangkan dengan set summary
    for sentence in scores.keys():
        if sentence not in summarySet:
            mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence, summarySet)   # 公式
    selected = max(mmr.items(), key=operator.itemgetter(1))[0]
    summarySet.append(selected)
    # print(summarySet)
    n -= 1

# print(str(time.time() - start))

print('Summary:')
rows = []
for sentence in summarySet:
    row = originalSentenceOf[sentence].lstrip(' ')
    rows.append(row)
summary = '。'.join(rows) + '。'
print(summary)

