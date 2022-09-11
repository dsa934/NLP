# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-29


< FastText >

 - ������ Word2Vec�� ��� �ּ� ������ �ܾ�� �����Ͽ� ��ūȭ�� �����ϴµ�,

   �� ��� �𸣴� �ܾ�(Out of Vocabulary, OOV)�� ���� ������ �����ϴٴ� ������ ���� �ִ�.

   �ݸ鿡, FastText�� �ܾ �� ���� �ܾ�� �ɰ� �� �ִٴ� ���̵� �����Ͽ�

   ������ Word2Vec�� ���� OOV , Rare Word�� ���� ������ �����ϴ� 


   Word_embedding_01 ������ ��, ������ ��ó�� ����� ������ train_data ���


<Step>

1. FastText �н�

   1-1. ������ �н� �� �� �ҷ�����

   1-2. ������ �н��� �𵨿� ����(OOV), ���� �ܾ �����Ͽ� ��� Ȯ���ϱ�

   1-3. ���� �ܾ ���� Fast Text �����ϱ�




'''

from gensim.models import FastText
from gensim.models import KeyedVectors


# Step 1 :  FastText �н�

# 1-1 : ���� �н��� �� �ҷ�����(word_embedding_01)
model = KeyedVectors.load_word2vec_format("eng_w2v")

# 1-2 : OOV, ���� �ܾ ���� ���� ��� Ȯ���ϱ�

# Failed Example
print(model.most_similar("electrofishing"))

# Success Example
print(model.most_similar("man"))

# 1-3 : FastText�� ������ �ܾ ���� 
fast_model = FastText(train_data, vector_size=100, window=5, min_count=5, workers=4, sg=1)

print(model.wv.most_similar("electrofishing"))