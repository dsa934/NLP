# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< Word2Vec �ѱ� Ver.  >

 - ���̹� ��ȭ ���� �����ͷ� �ѱ��� Word2Vec �����ϱ� 



<Step>

1. Train data ����

   1-1. ������ �ٿ�ε� 

   1-2. ������ ���� �� ��ūȭ

      => ����ġ(Null value) Ȯ�� 

         ���� ǥ������ ���� �ѱ� �� ���� ����

         �ҿ�� ����





2. �� ���� �� �н�

   2-1. �� ���� 

   2-2. ������ ���� Word2Vec Check




'''

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

# ������ �𸣰����� �Ʒ��� ���� ���·� �ۼ��ؾ� ���α׷����� ���� ����
from tqdm import tqdm


# Step 1

# 1-1 : ������ �ٿ�ε�
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')

# ���� 5���� ���� Ȯ�� 
print(train_data[:5])


# 1-2 : ������ ���� �� ��ūȭ

# ����ġ (Null) Ȯ��
if train_data.isnull().values.any() : 

    # Null ���� �����ϴ� row ���� 
    train_data = train_data.dropna(how = 'any')

# �ѱ� �� ���� ���� with ����ǥ����
train_data['document'] = train_data['document'].str.replace("[^��-����-�Ӱ�-�R ]","")

# �ҿ�� ����
not_using_words = ['��','��','��','��','��','��','��','��','��','��','��','��','����','��','��','��','��','�ϴ�']

# ���¼� �м��� OKT�� ����� ��ūȭ �۾� (�ټ� �ð� �ҿ�)
okt = Okt()

tokenized_data = []


# tqdm : ����� Progress bar 
for sentence in tqdm(train_data['document']):
    
    # ��ūȭ
    tokenized_sentence = okt.morphs(sentence, stem=True) 

    removed_sentence = [word for word in tokenized_sentence if not word in not_using_words]

    tokenized_data.append(removed_sentence)

# ���� ���� ���� Ȯ��
print('������ �ִ� ���� :',max(len(review) for review in tokenized_data))
print('������ ��� ���� :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# Step 2

# 2-1 �� ���� 

from gensim.models import Word2Vec

# model ����
# size -> vector size�� �ۼ��ؾ� �����ȳ� 
model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

# example 1
print(model.wv.most_similar("�ֹν�"))

print("===========")

# example 2
print(model.wv.most_similar("�����"))