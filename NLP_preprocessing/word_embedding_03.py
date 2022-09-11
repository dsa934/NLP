# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-29


< FastText >

 - 기존의 Word2Vec의 경우 최소 단위를 단어로 가정하여 토큰화를 적용하는데,

   이 경우 모르는 단어(Out of Vocabulary, OOV)에 대한 대응이 미흡하다는 단점을 갖고 있다.

   반면에, FastText는 단어를 더 작은 단어로 쪼갤 수 있다는 아이디어를 적용하여

   기존의 Word2Vec에 비해 OOV , Rare Word에 관한 대응이 가능하다 


   Word_embedding_01 에서의 모델, 데이터 전처리 기법을 적용한 train_data 사용


<Step>

1. FastText 학습

   1-1. 이전에 학습 된 모델 불러오기

   1-2. 이전에 학습한 모델에 실패(OOV), 성공 단어를 적용하여 결과 확인하기

   1-3. 실패 단어에 대해 Fast Text 적용하기




'''

from gensim.models import FastText
from gensim.models import KeyedVectors


# Step 1 :  FastText 학습

# 1-1 : 이전 학습된 모델 불러오기(word_embedding_01)
model = KeyedVectors.load_word2vec_format("eng_w2v")

# 1-2 : OOV, 성공 단어에 대한 적용 결과 확인하기

# Failed Example
print(model.most_similar("electrofishing"))

# Success Example
print(model.most_similar("man"))

# 1-3 : FastText를 실패한 단어에 적용 
fast_model = FastText(train_data, vector_size=100, window=5, min_count=5, workers=4, sg=1)

print(model.wv.most_similar("electrofishing"))