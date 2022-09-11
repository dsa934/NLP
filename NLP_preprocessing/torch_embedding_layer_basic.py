# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< nn.Embedding() >

 - pytorch�� nn.Embedding() ���� �ľ� �� �ǽ� �غ��� 


<Step>

1. nn.embedding() �Լ� ���� ���� �����غ��� 

   - ���� �� �ܾ embeding vector�� ��ȯ�ϱ� ���ؼ��� word�� integer embedding or one hot vector ȭ ���� 

     lookup table�� �ش��ϴ� Weight matrix�� �� index�� mapping �Ǵ� ��Ҹ� �о�;� �Ѵ�.

     ��, ������ word�� ���� ��ġ�� �ʰ� (����� �ǹ̰� �ƴ϶��) Ư���� ��ȣ or index�� �ο��޾ƾ� �Ѵ�.


2. nn.embedding() ����Ͽ� �����ϱ� 


'''

import torch

# Step 1

text_data = "you need to know how to code"

# ������ ��ĭ �������� ������, �ߺ��� �����Ͽ� �ܾ� ���� ����
text_set = set(text_data.split())

# �ܾ����� ���� 
vocabulary = {word : idx+2 for idx, word in enumerate(text_set)}

# unk := ���� �Է����� �޴� ������ �ܾ� ���տ� ���� ��츦 ���
vocabulary['<unk>'] = 0

# padding �� �ʿ��� ��츦 ��� ( library ��� ��)
vocabulary["<pad>"] = 1

# �Ӻ��� ���̺� ���� ( row = �ܾ������� ũ��, col = ����� ����(3) )
# size = 7 x 3 
# ���̺��� weight ���� ����ڰ� ����
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

# ������ ���忡 ���� embeding vector �����ϱ�
random_sentence = "you need to run".split()

embedding_idx = [] 

for word in random_sentence:

    # ������ ���带 �����ϴ� �ܾ �ܾ� ���տ� �ִ� ���
    if word in vocabulary.keys():

        embedding_idx.append(vocabulary[word])

    else:

        embedding_idx.append(vocabulary['<unk>'])
        
# 1D Tensor ( 1 x  len(list(randome_sentence.split())))
embedding_idx = torch.LongTensor(embedding_idx)

# result size = 4 x 3 
# ���� sliceing�� �ϸ� list[a:b]  a~ b-1 ���� ���Ҹ� �ǹ� 
# a= [1,2,3,4] ,  list[a,:] => a ����Ʈ�� ���� ���Ҹ� ������ ���, �ش� ���� ��� ����(:) �� �ǹ� 
result = embedding_table[embedding_idx,:]

print("Make Embedding vector without library")
print("sentence : you need to run  ")
print(result)



# Step 2

import torch.nn as nn
# num_embeddings := embedding table�� row ũ��
# embedding_dim := ����� ���� ����, colum ũ��
# padding_Idx := �е��� �ʿ��� ��� �е��� ���� token index �˷��� 
embeding_layer_with_library = nn.Embedding(num_embeddings = len(vocabulary), embedding_dim = 3, padding_idx = 1 )

print("show embedding table with library")
print(embeding_layer_with_library.weight)




