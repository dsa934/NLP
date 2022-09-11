# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< nn.Embedding() >

 - pytorch의 nn.Embedding() 원리 파악 및 실습 해보기 


<Step>

1. nn.embedding() 함수 없이 직접 구현해보기 

   - 문장 및 단어를 embeding vector로 변환하기 위해서는 word를 integer embedding or one hot vector 화 시켜 

     lookup table에 해당하는 Weight matrix의 각 index에 mapping 되는 요소를 읽어와야 한다.

     즉, 각각의 word는 서로 겹치지 않게 (비슷한 의미가 아니라면) 특정한 번호 or index를 부여받아야 한다.


2. nn.embedding() 사용하여 구현하기 


'''

import torch

# Step 1

text_data = "you need to know how to code"

# 문장을 빈칸 기준으로 나누고, 중복을 제거하여 단어 집합 형성
text_set = set(text_data.split())

# 단어집합 형성 
vocabulary = {word : idx+2 for idx, word in enumerate(text_set)}

# unk := 추후 입력으로 받는 문장이 단어 집합에 없는 경우를 대비
vocabulary['<unk>'] = 0

# padding 이 필요한 경우를 대비 ( library 사용 시)
vocabulary["<pad>"] = 1

# 임베딩 테이블 형성 ( row = 단어집합의 크기, col = 사용자 정의(3) )
# size = 7 x 3 
# 테이블의 weight 값을 사용자가 정함
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

# 임의의 문장에 대한 embeding vector 형성하기
random_sentence = "you need to run".split()

embedding_idx = [] 

for word in random_sentence:

    # 임의의 문장를 구성하는 단어가 단어 집합에 있는 경우
    if word in vocabulary.keys():

        embedding_idx.append(vocabulary[word])

    else:

        embedding_idx.append(vocabulary['<unk>'])
        
# 1D Tensor ( 1 x  len(list(randome_sentence.split())))
embedding_idx = torch.LongTensor(embedding_idx)

# result size = 4 x 3 
# 보통 sliceing을 하면 list[a:b]  a~ b-1 까지 원소를 의미 
# a= [1,2,3,4] ,  list[a,:] => a 리스트에 속한 원소를 행으로 잡고, 해당 행의 모든 원소(:) 를 의미 
result = embedding_table[embedding_idx,:]

print("Make Embedding vector without library")
print("sentence : you need to run  ")
print(result)



# Step 2

import torch.nn as nn
# num_embeddings := embedding table의 row 크기
# embedding_dim := 사용자 정의 차원, colum 크기
# padding_Idx := 패딩이 필요한 경우 패딩을 위한 token index 알려줌 
embeding_layer_with_library = nn.Embedding(num_embeddings = len(vocabulary), embedding_dim = 3, padding_idx = 1 )

print("show embedding table with library")
print(embeding_layer_with_library.weight)




