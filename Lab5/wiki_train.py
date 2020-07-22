from gensim.models import word2vec
import jieba
import logging 

def segment(sen_raw) -> list: # 对一个句子分词并返回一个链表
    sen = []
    try:
        sen = jieba.lcut(sen_raw)
    except:
        pass
    return sen

sub = [line.strip('\n') for line in open('zhwiki_500_processed.txt', 'r', errors = 'ignore').readlines()]
sens = [segment(i) for i in sub[:600000]]
# 生成日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO) 
model = word2vec.Word2Vec(sens, min_count=2,iter=25, window = 5, size = 500)
model.save("wiki_CBOW_w5_s500_m2.model")
model = word2vec.Word2Vec(sens, min_count=2,iter=25, window = 5, size = 500, sg = 1)
model.save("wiki_SKIP_GRAM_w5_s500_m2.model")