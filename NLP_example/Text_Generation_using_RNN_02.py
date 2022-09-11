# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-27


< Char RNN ( More Data , N:M RNN 구조를 이용한 ) >

 - 이전보다 더 많은 데이터에 대해서 문자 생성기 구현

   단순히 apple -> pple! 의 형태가 아니라
   
   문장이 주어지면, 해당 문장에 대해 스스로 입력, label을 형성하여 RNN 모델에 적용하기 



<Step>

1. train data preprocessing

   1-1. 문장 -> 글자 분리

   1-2. 샘플 데이터 구성

   1-3. one hot encoding

   1-4. 텐서화


2. Construct Model

  2-1. 2 hidden layers RNN 생성

  2-2. nn 선언 및 loss, optim 선언 

  2-3. train

'''


import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

# Step 1

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")


# 1-1
# 문장 -> 글자 분리  : set() 함수를 통해 문자열 전체를 알파벳 단위로 분리 
# sentecne.split() 사용시 단어 단위 분리가 되어버림 
# integer embeddings

alpha_list = list(set(sentence))

vocabulary = {word : idx for idx , word in enumerate( alpha_list) } 


print(vocabulary)

# model params
input_dim = len(vocabulary)
hidden_dim = input_dim
output_dim = input_dim
learning_rate = 0.1 


# 1-2
# make sample data
# input 0 : if you wan -> f you want 
# input 1 : f you want -> you want
# ...

x_train, y_label, seq_length = [] , [] , 10

for index in range(0, len(sentence)-seq_length):

    train_string = sentence[index:index+seq_length]
    label_string = sentence[index+1:index+1 + seq_length]

    # make sample
    x_train.append([vocabulary[char] for char in train_string])
    y_label.append([vocabulary[char] for char in label_string])


# 1-3
# one-hot encoding
train_data = [np.eye(len(vocabulary))[value] for value in x_train]

# 1-4
# 텐서화
# train_shape = [170, 10, 25] 
# test_shape = [170, 10]
train_data = torch.FloatTensor(train_data)
test_data = torch.LongTensor(y_label)



# Step 2

# 2-1
# Construct RNN with 2 hidden layers
class JW_RNN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, layers ) :

        super(JW_RNN,self).__init__()

        self.rnn = torch.nn.RNN(input_dim, hidden_dim, num_layers = layers, batch_first = True)

        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias = True )


    def forward(self, x):

        x, _status = self.rnn(x)

        x = self.fc(x)

        return x

 # 2-2
 # NN 선언
net = JW_RNN(len(vocabulary), hidden_dim, 2)

# Loss 기준 및 optimizer 선언
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

# 2-3 
# 학습 시작

for epoch in range(100):

    # optimizer 초기화
    optimizer.zero_grad()

    # prediction shpae = [170, 10, 25]
    output = net(train_data)

    loss = criterion( output.view(-1, len(vocabulary)) , test_data.view(-1) )

    # Backpropagation
    loss.backward()

    optimizer.step()

    # [170, 10, 25] = [batch, seq_len, 알파벳 종류 수 ] 임으로
    # 25개 중에 확률이 가장 큰 argmax(dim=2) 값이 다음 예측값일 확률이 높음
    predictions = output.argmax(dim=2)
    
    prediction_string = ""

    
    for j , prediction in enumerate(predictions):
        
        # 최초 예측
        if j == 0 :
            
            prediction_string += ''.join( [alpha_list[value] for value in prediction] )

        # 그 이후 예측 붙어는 seq data임으로 마지막 값만 가져오면 됨 
        else:

            prediction_string += alpha_list[prediction[-1]]

    
    print("epoch :" , epoch)
    print("=>", prediction_string)

