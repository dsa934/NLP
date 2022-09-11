# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< Text Preprocessing (Basic) >

 - NLP의 workflow는 

   데이터 수집 -> 데이터 분석 -> 데이터 전처리 -> 학습 -> 평가 -> 배포 순으로 이루어진다.

   이 중 데이터 전처리 과정(text preprocessing)의 여러 tokenization 기법을 직접 구현해보면서 이론 및 실전 설계 지식을 습득 하는것을 목적



<Step>

1. nltk library의 tokenization 활용하기 

2. konpy.tag의 형태소 분석기 Okt 활용하기 

'''


# Step 1
from nltk.tokenize import word_tokenize
print("tokenization through nltk library\n")
print("sentence : Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop. \n")
print("=>", word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))



# Step 2 
from konlpy.tag import Okt

def one_hot_encoding(word, vocabulary):

    one_hot_vec = [0] * (len(vocabulary)+1)
    
    one_hot_vec[vocabulary[word]] = 1

    print("one-hot vector, word : ",  one_hot_vec, word)


   

def tokenize():

    okt = Okt()

    # 형태소 분석기를 통한 문장 토큰화 
    sentense = okt.morphs("자연어 처리 학습을 위한 텍스트 전처리과정 예습중입니다.")

    # 형태소 분리 체크 
    print(sentense)

    vocabulary = {} 

    for index, word in enumerate(sentense):

        if word not in vocabulary.keys():

            vocabulary[word] = index
        
    # one-hot vector를 위한 단어 집합 확인 
    print("단어 집합 : " , vocabulary)


    # 각 단어별 one-hot vector 출력
    for word in vocabulary.keys():
        
        one_hot_encoding(word, vocabulary)


tokenize()


