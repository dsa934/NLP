# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-29


< ���� ���͸� �̿��� ��õ �ý��� >

 - ������ �� �ܾ���� Word2Vec�� ���� embedding vector�� ��ȯ�Ͽ�, �̵��� ��հ��� ���� embedding vector value��� ����

   �� ���� �� ���絵 ��� ����


   => embedding�� �� �� ���, �ܾ� ������ ��ո����� �ؽ�Ʈ �з��� ���� ��������


   <2022-8-29 ���� �̿ϼ� ���� ���� �䱸 > 

<Step>

1. semi_title

   - contents


'''

import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

urllib.request.urlretrieve("https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/09.%20Word%20Embedding/dataset/data.csv", filename="data.csv")
df = pd.read_csv("data.csv")
print('��ü ������ �� :',len(df))

def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

def make_lower_case(text):
    return text.lower()

def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

df['cleaned'] = df['Desc'].apply(_removeNonAscii)
df['cleaned'] = df.cleaned.apply(make_lower_case)
df['cleaned'] = df.cleaned.apply(remove_stop_words)
df['cleaned'] = df.cleaned.apply(remove_punctuation)
df['cleaned'] = df.cleaned.apply(remove_html)

df['cleaned'].replace('', np.nan, inplace=True)
df = df[df['cleaned'].notna()]
print('��ü ������ �� :',len(df))

corpus = []
for words in df['cleaned']:
    corpus.append(words.split())


# Step 2 : ���� �Ʒõ� Word2Vec ����
urllib.request.urlretrieve("https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz", \
                           filename="GoogleNews-vectors-negative300.bin.gz")

word2vec_model = Word2Vec(size = 300, window=5, min_count = 2, workers = -1)
word2vec_model.build_vocab(corpus)
word2vec_model.intersect_word2vec_format('GoogleNews-vectors-negative300.bin.gz', lockf=1.0, binary=True)
word2vec_model.train(corpus, total_examples = word2vec_model.corpus_count, epochs = 15)

# Step 3 : ���� embedding vector ���ϱ� (�ܾ� ������ ��� ���ϱ� )
def get_document_vectors(document_list):
    document_embedding_list = []

    # �� ������ ���ؼ�
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word in word2vec_model.wv.vocab:
                count += 1
                # �ش� ������ �ִ� ��� �ܾ���� ���Ͱ��� ���Ѵ�.
                if doc2vec is None:
                    doc2vec = word2vec_model[word]
                else:
                    doc2vec = doc2vec + word2vec_model[word]

        if doc2vec is not None:
            # �ܾ� ���͸� ��� ���� ������ ���� ���� ���̷� �����ش�.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)

    # �� ������ ���� ���� ���� ����Ʈ�� ����
    return document_embedding_list

document_embedding_list = get_document_vectors(df['cleaned'])
print('���� ������ �� :',len(document_embedding_list))

# Step 4 : ��õ �ý��� �����ϱ�

cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)
print('�ڻ��� ���絵 ��Ʈ������ ũ�� :',cosine_similarities.shape)

def recommendations(title):
    books = df[['title', 'image_link']]

    # å�� ������ �Է��ϸ� �ش� ������ �ε����� ���Ϲ޾� idx�� ����.
    indices = pd.Series(df.index, index = df['title']).drop_duplicates()    
    idx = indices[title]

    # �Էµ� å�� �ٰŸ�(document embedding)�� ������ å 5�� ����.
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]

    # ���� ������ å 5���� �ε���
    book_indices = [i[0] for i in sim_scores]

    # ��ü �����������ӿ��� �ش� �ε����� �ุ ����. 5���� ���� ������.
    recommend = books.iloc[book_indices].reset_index(drop=True)

    fig = plt.figure(figsize=(20, 30))

    # ���������������κ��� ���������� �̹����� ���
    for index, row in recommend.iterrows():
        response = requests.get(row['image_link'])
        img = Image.open(BytesIO(response.content))
        fig.add_subplot(1, 5, index + 1)
        plt.imshow(img)
        plt.title(row['title'])


recommendations("The Da Vinci Code")