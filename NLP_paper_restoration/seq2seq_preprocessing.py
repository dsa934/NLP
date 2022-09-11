# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-01


< seq2seq preprocessing >

  1. 영어, 독일어 전처리 모듈 설치 ( Anaconda Ver.) 및 로드 

     python -m spacy download en
     python -m spacy download de


  2. 데이터에 적용되는 Field 설정

     train, label 데이터에 대하여 < 어떤 방식의 토큰화, 시작토큰, 종료토큰, 소문자 변환여부 > 등을 설정한 Field 객체 설정

     필드 객체 설정 이후 , 실제 데이터를 매핑하면 실 데이터가 field 객체 형태로 정리되는 형태


  3. dataset을 이용한 실제 데이터 로드 및 field 적용 

     torchtext.dataset의 경우 IMDB(리뷰데이터), Multk30k 와 같은 특정 데이터셋 로드

     <dataset>.splits( exts =( a from b) , fields = (train, test) ) 함수를 통해 

     train, val, test dataset 분할 후, 입력 데이터 형태(a) 와 정답 데이터 형태(b) 를 설정하여 각 필드 객체에 적용 


  4. 단어 집합( Vocabulary ) 설정

    embedding layer에 입력 데이터로 활용 되기 위해서는 토큰화 된 단어(문자)에 대한 정수 임베딩 값이 필요 

    중복을 제외한 모든 단어에 대한 정수 임베딩이 필요하기 때문에 train, label에 대한 전체 단어 집합의 수 요구 



  5. BucketIterator를 이용한 mini batch

     전체 데이터 셋을 train, val, test 데이터의 형태로 나눈 후

     각각의 데이터를 batch_size만큼 분할 

     train_data = 29,000 , batch_size = 128 => 29000/128 = 226.5 
     
     즉,  226 full iterators + 1 deficient iterator( 1개의 iteration이 78개 ) = 227 iterators


     val_data = 1,014, batch_size = 128 = > 1014/128 = 7.9 

     즉, 7 ful iterators + 1 deficient iterator = 7 iterators



  cf : 구현시 참고 

   * torchtext issue 

     => torchtext 로는 data, datasets이 접근 되지 않아 torchtext.legacy를 사용하라고 요구되는데,

        더 나은 방법은 아래와 같은 방법이 사용 됨 

        from torchtext.data import <원하는 함수>

        from torchtext.dataset import <원하는 데이터셋> 


   * cuda devcie set

     => device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


   * BucketIterator로 인해 균등 분할 된 train, val, test iterator는 

     for idx, batch in enumerate(train_iter) 와 같이 사용 될 수 있는데

     이때 데이터에 해당하는 batch의 구성은 

     x_train, x_val, y_test = Multi30k.splits(exts=(".de", ".en"), fields=(train,label))

     에서 설정한 field 값으로 구성 된다 


     ** 하지만 여러번 테스트 해본 결과 Multi30k 데이터의 경우 이미 내부적으로 field가 src, trg로 셋팅 되 있는것 같다

        (train,label)과 같은 다른 이름 쌍으로 field 객체를 구현하고 적용해도, src, trg 로 구성되는것을 확인할 수 있었다.



     
        

'''

import spacy

# 영어 & 불어(de) 토큰화를 위한 전처리 라이브러리 로드 
en_token = spacy.load('en_core_web_sm')
de_token = spacy.load('de_core_news_sm')

# 불어 -> 영어 번역 임으로, 불어가 입력
# seq2seq 논문에 의하면 입력 text의 순서를 뒤집는 것이 더 성능이 높다
def de_token_fuc(text):

    return [ _token.text for _token in de_token(text)][::-1]


def en_token_fuc(text):
    
    return [ _token.text for _token in en_token(text) ]


# torchtext issue
from torchtext.data import Field, BucketIterator

# 데이터에 적용할 field 를 설정
src = Field(tokenize = de_token_fuc, init_token="<sos>", eos_token="<eos>", lower = True )
trg = Field(tokenize = en_token_fuc, init_token = "<sos>", eos_token = "<eos>", lower = True )


# dataset을 이용한 실제 데이터 로드 및 field 적용 
from torchtext.datasets import Multi30k
x_train, x_val, y_test = Multi30k.splits(exts=(".de", ".en"), fields=(src,trg))

# result =>  train(29,000), val(1,014),  test(1,000)
print(f" train data size :  {len(x_train.examples)}")
print(f" val data size :  {len(x_val.examples)}")
print(f" test data size :  {len(y_test.examples)}")


# 단어집합 형성
# len(src)= 7855 > len(trg):5893
# 불어가 영어보다 각 문장에 대한 단어 구성이 많음을 의미 
src.build_vocab(x_train, min_freq = 2)
trg.build_vocab(x_train, min_freq = 2 )

# token 확인
# <unk> : 0 ,  <padding> : 1,  <sos> : 2 , < eos> : 3 
print(trg.vocab.stoi["<sos>"])
print(trg.vocab.stoi["<eos>"])
print(trg.vocab.stoi[trg.pad_token])
print(trg.vocab.stoi["hello"])
print(trg.vocab.stoi["없을것만 같은 단어"])


import torch

# current status : cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, val_iter, test_iter = BucketIterator.splits((x_train, x_val, y_test), batch_size = 128, device = device)

print(f" train iterator size :  {len(train_iter)}")
print(f" val iterator size :  {len(val_iter)}")
print(f" test iterator size :  {len(test_iter)}")


# train data 확인
for idx, batch in enumerate(train_iter):
    
    chk_train = batch.src
    chk_label = batch.trg

    # 첫 배치에 있는 데이터 체크
    for data_idx, value in enumerate(chk_train[1]):

        print(f"idex : {data_idx} : {value}")
        

    break


