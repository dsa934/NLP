# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-27


< Char RNN ( N:M RNN 구조를 이용한 ) >

 -  apple 를 입력받아 pple!를 출력하는 RNN text generator 구현을 통해 RNN의 동작방식 알아보기 

    N:N (다대다) RNN을 이용한 텍스트 생성 ( without embedding layer )

    many to many 1번 유형 ( 기계 번역 처럼 encoder, decoder 구분 된 것이 아닌, 매 시점 입력에 대한 출력이 존재 )

     O  0  0
     |  |  |
     ㅁ ㅁ ㅁ
     |  |  |
     0  0  0


<Step>

1. train data preprocessing


2. Construct Model

'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Step 1 : Train data preprocesing

input_string = "apple"
label = "pple!"

# 단어집합(vocabulary) 형성
# {a, p, l ,e , !}
vocabulary = sorted(list(set(input_string + label)))

# integer embedding
char_embedding = { char: idx for idx, char in enumerate(vocabulary) }

# integer to word 
# training 이후 , 학습 과정을 알아보기 위해 복원하는 과정에서 씌임 
number_to_word = {idx : char for idx, char in enumerate(vocabulary) }

# 입력 데이터 정수 임베딩 [1,4,4,3,2]
train_data = [ char_embedding[char] for char in input_string ]

# 출력 데이터 정수 임베딩 [4,4,3,2,0]
label_data = [ char_embedding[char] for char in label ]

# RNN의 입력은 batch를 고려한 3D Tensor 이며,
# Tensor의 형태는 [ batch, row, col] 의 형태를 갖기 떄문에 
# 입력 데이터 및 정답 데이터를 3D tensor 화 
train_data, label_data = [train_data], [label_data]

# one hot vector 화 
# embedding layer를 사용하지 않기 떄문에 정수 임베딩 된 데이터를 
# 해당 정수에 맞는 one-hot vector로 변환

# np.eye(size) : size 크기의 대각 행렬 형성 
# 크기가 5인 대각행렬은 [1,0,0,0,0] , [0,1,0,0,0] ... [0,0,0,0,1] 의 형태로 구성 되있으므로
# train_data에 각 문자에 대한 정수 임베딩 값 위치의 대각행렬을 가져오게 됨으로 one-hot vector를 의미 
one_hot_train_data = [ np.eye(len(vocabulary))[value] for value in train_data]

# vector의 tensor화 
x_train = torch.FloatTensor(one_hot_train_data)
y_test = torch.LongTensor(label_data)

# tensor dimension chk
# x_train = [1,5,5] , y_train = [1,5]
print(x_train.shape)
print(y_test.shape)



# Step 2 : Construct RNN model

# model params
# many to many first case로써 매 시점 입력에 대한 출력이 형성 됨으로
# 입력과 출력의 크기가 같아야 한다
input_size = len(vocabulary)
hidden_size = 5
output_size = 5
learning_rate = 0.1

# RNN 구조
#
#
#  (output area with FC)      output
#  ---------------------------  ↑
#  (RNN cell area)
#                h_(t-1)  ->   h_(t)
#                               ↑
#                              input

class Customized_RNN(torch.nn.Module):

    def __init__(self, input_size , hidden_size, output_size):

        super(Customized_RNN, self).__init__()

        # RNN cell 구축 ( RNN cell area)
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first = True)

        # output area with FC
        self.fc = torch.nn.Linear(hidden_size, output_size, bias=True)


    def forward(self, x):

        x, _status = self.rnn(x)

        x = self.fc(x)

        return x


# create RNN
net = Customized_RNN(input_size, hidden_size, output_size)

# set loss criterion & optimizer type
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)


for epoch in range(100):

    # optim init
    optimizer.zero_grad()

    # x_train.shape = [1,5,5]
    # after net , output.shape = [1,5,5]
    output = net(x_train)

    # output.view(-1, input_size)  , batch 차원 제거를 위해 사용 ( 정확도 측정 시, batch_size만큼 나눠진 모든 데이터를 합쳐서 정확도를 계산함으로 3D -> 2D Tensor 화 )
    # data.view(-1, size) : data의 원래 차원 구성과 무관하게 [?, size] 형태로 변환
    # example) If  data = [ 2,3,5] , size = 5 then  data.view[-1,5]  ->  data.shape = [6, 5] 
    loss = criterion( output.view(-1, input_size), y_test.view(-1) )
    
    # 기울기 계산
    loss.backward()

    # weight update
    optimizer.step()

    # 실제 어떻게 예측했는지 보여주기 위한 code

    # 최종 예측값 에 해당하는 각 time_step 별 5차원 vector에 대해 가장 높은 값의 index 선택 
    result = output.data.numpy().argmax(axis=2)
    
    # result.shape = (1,5) = [1,1,5] 형태라 np.squeeze()를 이용하여 가장 바깥 차원 제거(제일 왼쪽)
    # ex) [[4 4 0 4 0 ]] => [4 4 0 4 0]
    result_string = ''.join( [number_to_word[value] for value in np.squeeze(result) ])  

    # loss value 만 추출할떈 loss.item()
    print("Epoch : ", epoch , "Loss : " , loss.item(), "Prediction : ", result, "Label : ", y_test, "Prediction string :", result_string)





