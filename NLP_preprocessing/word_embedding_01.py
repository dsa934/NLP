# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< 영어 Word2Vec 만들기  >

 - gensim pkgs 의 Word2Vec을 활용하여 영어 Word2Vec 만들어보기 



<Step>

1. 훈련 데이터 구축하기

   1-1. 데이터 다운로드 및 전처리

        => xml 데이터를 파싱 후 , 파싱 된 데이터에서 필요한 특정 부분에 대한 데이터만 읽어와

           join 함수를 통해 하나의 문자열로 합친다 이 떄 각 문장의 구분을 위해 \n을 사용 ( '\n'.join(data))

           정규 표현식을 통해 불필요한 내용(배경음, 괄호로 구성된 내용) 제거


   1-2.  문장 토큰화 

        => xml 파일에서 여러개의 문장을 하나의 문자열로 구성했음으로, 문장에 대한 tokenization 진행 

           구두점 제거, 대문자 -> 소문자 변환 


   1-3.  단어 토큰화



2. Word2Vec Training

   2-1. 모델 설정 후 다뤄보기

       =>  학습 

           저장 & 불러오기 




'''


import re
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import urllib.request


# Step 1 

# 1-1 : 데이터 다운로드 & 전처리
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

print("done Download")

# xml 데이터 파일 열어서 파싱
_parse = etree.parse( open('ted_en-20160408.xml', 'r', encoding='UTF8') )

# xml 파일의 contents 부분 내용만 가져옴 
# '\n'.join(data) : parsing 되어 토큰으로 나뉘어진 데이터에 대하여, \n을 붙여서 하나의 문자열로 구성
_text = "\n".join(_parse.xpath('//content/text()'))

# 정규 표현식을 통해 content에 속한 배경음, 괄호로 구성된 내용 제거
text = re.sub(r'\([^)]*\)', '', _text)

print("done Parsing")

# 1-2 : sentence tokenizatoin ( 다소 시간 많이 소모)
sentence = sent_tokenize(text)

normalized_sentence_set = []

for string in sentence:

    print("Str:",string)

    norm_sentence = re.sub(r"[^a-z0-9]+", " ", string.lower())

    normalized_sentence_set.append(norm_sentence)

print("done normalized")

# 1-3 : word tokenization
train_data = [ word_tokenize(_sentence) for _sentence in normalized_sentence_set ]

print('총 샘플의 개수 : {}'.format(len(train_data)))

print("Start model training")

# Step 2

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 2-1 : Word2Vec training
# size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
# window = 컨텍스트 윈도우 크기
# min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
# workers = 학습을 위한 프로세스 수
# sg = 0은 CBOW, 1은 Skip-gram.
model = Word2Vec(sentences = train_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)

# training result
model_result = model.wv.most_similar("man")
print(model_result)


# model save and load
model.wv.save_word2vec_format('eng_w2v')
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") 

