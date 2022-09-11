# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-01


< seq2seq preprocessing >

  1. ����, ���Ͼ� ��ó�� ��� ��ġ ( Anaconda Ver.) �� �ε� 

     python -m spacy download en
     python -m spacy download de


  2. �����Ϳ� ����Ǵ� Field ����

     train, label �����Ϳ� ���Ͽ� < � ����� ��ūȭ, ������ū, ������ū, �ҹ��� ��ȯ���� > ���� ������ Field ��ü ����

     �ʵ� ��ü ���� ���� , ���� �����͸� �����ϸ� �� �����Ͱ� field ��ü ���·� �����Ǵ� ����


  3. dataset�� �̿��� ���� ������ �ε� �� field ���� 

     torchtext.dataset�� ��� IMDB(���䵥����), Multk30k �� ���� Ư�� �����ͼ� �ε�

     <dataset>.splits( exts =( a from b) , fields = (train, test) ) �Լ��� ���� 

     train, val, test dataset ���� ��, �Է� ������ ����(a) �� ���� ������ ����(b) �� �����Ͽ� �� �ʵ� ��ü�� ���� 


  4. �ܾ� ����( Vocabulary ) ����

    embedding layer�� �Է� �����ͷ� Ȱ�� �Ǳ� ���ؼ��� ��ūȭ �� �ܾ�(����)�� ���� ���� �Ӻ��� ���� �ʿ� 

    �ߺ��� ������ ��� �ܾ ���� ���� �Ӻ����� �ʿ��ϱ� ������ train, label�� ���� ��ü �ܾ� ������ �� �䱸 



  5. BucketIterator�� �̿��� mini batch

     ��ü ������ ���� train, val, test �������� ���·� ���� ��

     ������ �����͸� batch_size��ŭ ���� 

     train_data = 29,000 , batch_size = 128 => 29000/128 = 226.5 
     
     ��,  226 full iterators + 1 deficient iterator( 1���� iteration�� 78�� ) = 227 iterators


     val_data = 1,014, batch_size = 128 = > 1014/128 = 7.9 

     ��, 7 ful iterators + 1 deficient iterator = 7 iterators



  cf : ������ ���� 

   * torchtext issue 

     => torchtext �δ� data, datasets�� ���� ���� �ʾ� torchtext.legacy�� ����϶�� �䱸�Ǵµ�,

        �� ���� ����� �Ʒ��� ���� ����� ��� �� 

        from torchtext.data import <���ϴ� �Լ�>

        from torchtext.dataset import <���ϴ� �����ͼ�> 


   * cuda devcie set

     => device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


   * BucketIterator�� ���� �յ� ���� �� train, val, test iterator�� 

     for idx, batch in enumerate(train_iter) �� ���� ��� �� �� �ִµ�

     �̶� �����Ϳ� �ش��ϴ� batch�� ������ 

     x_train, x_val, y_test = Multi30k.splits(exts=(".de", ".en"), fields=(train,label))

     ���� ������ field ������ ���� �ȴ� 


     ** ������ ������ �׽�Ʈ �غ� ��� Multi30k �������� ��� �̹� ���������� field�� src, trg�� ���� �� �ִ°� ����

        (train,label)�� ���� �ٸ� �̸� ������ field ��ü�� �����ϰ� �����ص�, src, trg �� �����Ǵ°��� Ȯ���� �� �־���.



     
        

'''

import spacy

# ���� & �Ҿ�(de) ��ūȭ�� ���� ��ó�� ���̺귯�� �ε� 
en_token = spacy.load('en_core_web_sm')
de_token = spacy.load('de_core_news_sm')

# �Ҿ� -> ���� ���� ������, �Ҿ �Է�
# seq2seq ���� ���ϸ� �Է� text�� ������ ������ ���� �� ������ ����
def de_token_fuc(text):

    return [ _token.text for _token in de_token(text)][::-1]


def en_token_fuc(text):
    
    return [ _token.text for _token in en_token(text) ]


# torchtext issue
from torchtext.data import Field, BucketIterator

# �����Ϳ� ������ field �� ����
src = Field(tokenize = de_token_fuc, init_token="<sos>", eos_token="<eos>", lower = True )
trg = Field(tokenize = en_token_fuc, init_token = "<sos>", eos_token = "<eos>", lower = True )


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
src.build_vocab(x_train, min_freq = 2)
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

print(f" train iterator size :  {len(train_iter)}")
print(f" val iterator size :  {len(val_iter)}")
print(f" test iterator size :  {len(test_iter)}")


# train data Ȯ��
for idx, batch in enumerate(train_iter):
    
    chk_train = batch.src
    chk_label = batch.trg

    # ù ��ġ�� �ִ� ������ üũ
    for data_idx, value in enumerate(chk_train[1]):

        print(f"idex : {data_idx} : {value}")
        

    break


