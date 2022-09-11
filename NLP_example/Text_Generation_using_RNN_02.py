# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-27


< Char RNN ( More Data , N:M RNN ������ �̿��� ) >

 - �������� �� ���� �����Ϳ� ���ؼ� ���� ������ ����

   �ܼ��� apple -> pple! �� ���°� �ƴ϶�
   
   ������ �־�����, �ش� ���忡 ���� ������ �Է�, label�� �����Ͽ� RNN �𵨿� �����ϱ� 



<Step>

1. train data preprocessing

   1-1. ���� -> ���� �и�

   1-2. ���� ������ ����

   1-3. one hot encoding

   1-4. �ټ�ȭ


2. Construct Model

  2-1. 2 hidden layers RNN ����

  2-2. nn ���� �� loss, optim ���� 

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
# ���� -> ���� �и�  : set() �Լ��� ���� ���ڿ� ��ü�� ���ĺ� ������ �и� 
# sentecne.split() ���� �ܾ� ���� �и��� �Ǿ���� 
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
# �ټ�ȭ
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
 # NN ����
net = JW_RNN(len(vocabulary), hidden_dim, 2)

# Loss ���� �� optimizer ����
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), learning_rate)

# 2-3 
# �н� ����

for epoch in range(100):

    # optimizer �ʱ�ȭ
    optimizer.zero_grad()

    # prediction shpae = [170, 10, 25]
    output = net(train_data)

    loss = criterion( output.view(-1, len(vocabulary)) , test_data.view(-1) )

    # Backpropagation
    loss.backward()

    optimizer.step()

    # [170, 10, 25] = [batch, seq_len, ���ĺ� ���� �� ] ������
    # 25�� �߿� Ȯ���� ���� ū argmax(dim=2) ���� ���� �������� Ȯ���� ����
    predictions = output.argmax(dim=2)
    
    prediction_string = ""

    
    for j , prediction in enumerate(predictions):
        
        # ���� ����
        if j == 0 :
            
            prediction_string += ''.join( [alpha_list[value] for value in prediction] )

        # �� ���� ���� �پ�� seq data������ ������ ���� �������� �� 
        else:

            prediction_string += alpha_list[prediction[-1]]

    
    print("epoch :" , epoch)
    print("=>", prediction_string)

