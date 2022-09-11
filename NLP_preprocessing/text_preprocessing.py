# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< Text Preprocessing (Basic) >

 - NLP�� workflow�� 

   ������ ���� -> ������ �м� -> ������ ��ó�� -> �н� -> �� -> ���� ������ �̷������.

   �� �� ������ ��ó�� ����(text preprocessing)�� ���� tokenization ����� ���� �����غ��鼭 �̷� �� ���� ���� ������ ���� �ϴ°��� ����



<Step>

1. nltk library�� tokenization Ȱ���ϱ� 

2. konpy.tag�� ���¼� �м��� Okt Ȱ���ϱ� 

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

    # ���¼� �м��⸦ ���� ���� ��ūȭ 
    sentense = okt.morphs("�ڿ��� ó�� �н��� ���� �ؽ�Ʈ ��ó������ �������Դϴ�.")

    # ���¼� �и� üũ 
    print(sentense)

    vocabulary = {} 

    for index, word in enumerate(sentense):

        if word not in vocabulary.keys():

            vocabulary[word] = index
        
    # one-hot vector�� ���� �ܾ� ���� Ȯ�� 
    print("�ܾ� ���� : " , vocabulary)


    # �� �ܾ one-hot vector ���
    for word in vocabulary.keys():
        
        one_hot_encoding(word, vocabulary)


tokenize()


