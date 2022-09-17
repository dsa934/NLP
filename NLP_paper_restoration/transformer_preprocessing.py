# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-13


 * 기존 preprocessing 과 다른점은 

   tokenization 과정에서 batch_first = True 

'''


import spacy

# 영어 & 불어(de) 토큰화를 위한 전처리 라이브러리 로드 
en_token = spacy.load('en_core_web_sm')
de_token = spacy.load('de_core_news_sm')

# 불어 -> 영어 번역 임으로, 불어가 입력
def de_token_fuc(text):

    return [ _token.text for _token in de_token(text)]


def en_token_fuc(text):
    
    return [ _token.text for _token in en_token(text) ]


# torchtext issue
from torchtext.data import Field, BucketIterator

# 데이터에 적용할 field 를 설정
src = Field(tokenize = de_token_fuc, init_token="<sos>", eos_token="<eos>", lower = True , batch_first = True)
trg = Field(tokenize = en_token_fuc, init_token = "<sos>", eos_token = "<eos>", lower = True , batch_first = True)


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
src.build_vocab(x_train, min_freq = 2 )
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

