# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< ���� Word2Vec �����  >

 - gensim pkgs �� Word2Vec�� Ȱ���Ͽ� ���� Word2Vec ������ 



<Step>

1. �Ʒ� ������ �����ϱ�

   1-1. ������ �ٿ�ε� �� ��ó��

        => xml �����͸� �Ľ� �� , �Ľ� �� �����Ϳ��� �ʿ��� Ư�� �κп� ���� �����͸� �о��

           join �Լ��� ���� �ϳ��� ���ڿ��� ��ģ�� �� �� �� ������ ������ ���� \n�� ��� ( '\n'.join(data))

           ���� ǥ������ ���� ���ʿ��� ����(�����, ��ȣ�� ������ ����) ����


   1-2.  ���� ��ūȭ 

        => xml ���Ͽ��� �������� ������ �ϳ��� ���ڿ��� ������������, ���忡 ���� tokenization ���� 

           ������ ����, �빮�� -> �ҹ��� ��ȯ 


   1-3.  �ܾ� ��ūȭ



2. Word2Vec Training

   2-1. �� ���� �� �ٷﺸ��

       =>  �н� 

           ���� & �ҷ����� 




'''


import re
import zipfile
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import urllib.request


# Step 1 

# 1-1 : ������ �ٿ�ε� & ��ó��
urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/ted_en-20160408.xml", filename="ted_en-20160408.xml")

print("done Download")

# xml ������ ���� ��� �Ľ�
_parse = etree.parse( open('ted_en-20160408.xml', 'r', encoding='UTF8') )

# xml ������ contents �κ� ���븸 ������ 
# '\n'.join(data) : parsing �Ǿ� ��ū���� �������� �����Ϳ� ���Ͽ�, \n�� �ٿ��� �ϳ��� ���ڿ��� ����
_text = "\n".join(_parse.xpath('//content/text()'))

# ���� ǥ������ ���� content�� ���� �����, ��ȣ�� ������ ���� ����
text = re.sub(r'\([^)]*\)', '', _text)

print("done Parsing")

# 1-2 : sentence tokenizatoin ( �ټ� �ð� ���� �Ҹ�)
sentence = sent_tokenize(text)

normalized_sentence_set = []

for string in sentence:

    print("Str:",string)

    norm_sentence = re.sub(r"[^a-z0-9]+", " ", string.lower())

    normalized_sentence_set.append(norm_sentence)

print("done normalized")

# 1-3 : word tokenization
train_data = [ word_tokenize(_sentence) for _sentence in normalized_sentence_set ]

print('�� ������ ���� : {}'.format(len(train_data)))

print("Start model training")

# Step 2

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# 2-1 : Word2Vec training
# size = ���� ������ Ư¡ ��. ��, �Ӻ��� �� ������ ����.
# window = ���ؽ�Ʈ ������ ũ��
# min_count = �ܾ� �ּ� �� �� ���� (�󵵰� ���� �ܾ���� �н����� �ʴ´�.)
# workers = �н��� ���� ���μ��� ��
# sg = 0�� CBOW, 1�� Skip-gram.
model = Word2Vec(sentences = train_data, vector_size=100, window=5, min_count=5, workers=4, sg=0)

# training result
model_result = model.wv.most_similar("man")
print(model_result)


# model save and load
model.wv.save_word2vec_format('eng_w2v')
loaded_model = KeyedVectors.load_word2vec_format("eng_w2v") 

