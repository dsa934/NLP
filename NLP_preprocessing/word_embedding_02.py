# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< Word2Vec 한글 Ver.  >

 - 네이버 영화 리뷰 데이터로 한국어 Word2Vec 구축하기 



<Step>

1. Train data 구축

   1-1. 데이터 다운로드 

   1-2. 데이터 정제 및 토큰화

      => 결측치(Null value) 확인 

         정규 표현식을 통해 한글 외 문자 제거

         불용어 제거





2. 모델 구축 및 학습

   2-1. 모델 선언 

   2-2. 예제를 통한 Word2Vec Check




'''

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

# 왜인지 모르겠으나 아래와 같은 형태로 작성해야 프로그래스바 진행 가능
from tqdm import tqdm


# Step 1

# 1-1 : 데이터 다운로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('ratings.txt')

# 상위 5개의 리뷰 확인 
print(train_data[:5])


# 1-2 : 데이터 정제 및 토큰화

# 결측치 (Null) 확인
if train_data.isnull().values.any() : 

    # Null 값이 존재하는 row 제거 
    train_data = train_data.dropna(how = 'any')

# 한글 외 문자 제거 with 정규표현식
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 불용어 정의
not_using_words = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 형태소 분석기 OKT를 사용한 토큰화 작업 (다소 시간 소요)
okt = Okt()

tokenized_data = []


# tqdm : 진행률 Progress bar 
for sentence in tqdm(train_data['document']):
    
    # 토큰화
    tokenized_sentence = okt.morphs(sentence, stem=True) 

    removed_sentence = [word for word in tokenized_sentence if not word in not_using_words]

    tokenized_data.append(removed_sentence)

# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(review) for review in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(review) for review in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# Step 2

# 2-1 모델 구축 

from gensim.models import Word2Vec

# model 선언
# size -> vector size로 작성해야 오류안남 
model = Word2Vec(sentences = tokenized_data, vector_size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

# example 1
print(model.wv.most_similar("최민식"))

print("===========")

# example 2
print(model.wv.most_similar("히어로"))