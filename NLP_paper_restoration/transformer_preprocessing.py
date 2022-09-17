# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-13


 * ���� preprocessing �� �ٸ����� 

   tokenization �������� batch_first = True 

'''


import spacy

# ���� & �Ҿ�(de) ��ūȭ�� ���� ��ó�� ���̺귯�� �ε� 
en_token = spacy.load('en_core_web_sm')
de_token = spacy.load('de_core_news_sm')

# �Ҿ� -> ���� ���� ������, �Ҿ �Է�
def de_token_fuc(text):

    return [ _token.text for _token in de_token(text)]


def en_token_fuc(text):
    
    return [ _token.text for _token in en_token(text) ]


# torchtext issue
from torchtext.data import Field, BucketIterator

# �����Ϳ� ������ field �� ����
src = Field(tokenize = de_token_fuc, init_token="<sos>", eos_token="<eos>", lower = True , batch_first = True)
trg = Field(tokenize = en_token_fuc, init_token = "<sos>", eos_token = "<eos>", lower = True , batch_first = True)


# dataset�� �̿��� ���� ������ �ε� �� field ���� 
from torchtext.datasets import Multi30k
x_train, x_val, y_test = Multi30k.splits(exts=(".de", ".en"), fields=(src,trg))

# result =>  train(29,000), val(1,014),  test(1,000)
print(f" train data size :  {len(x_train.examples)}")
print(f" val data size :  {len(x_val.examples)}")
print(f" test data size :  {len(y_test.examples)}")


# �ܾ����� ����
# len(src)= 7855 > len(trg):5893
# �Ҿ ����� �� ���忡 ���� �ܾ� ������ ������ �ǹ� 
src.build_vocab(x_train, min_freq = 2 )
trg.build_vocab(x_train, min_freq = 2 )

# token Ȯ��
# <unk> : 0 ,  <padding> : 1,  <sos> : 2 , < eos> : 3 
print(trg.vocab.stoi["<sos>"])
print(trg.vocab.stoi["<eos>"])
print(trg.vocab.stoi[trg.pad_token])
print(trg.vocab.stoi["hello"])
print(trg.vocab.stoi["�����͸� ���� �ܾ�"])


import torch

# current status : cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, val_iter, test_iter = BucketIterator.splits((x_train, x_val, y_test), batch_size = 128, device = device)

