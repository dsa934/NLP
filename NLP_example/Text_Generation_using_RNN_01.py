# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-27


< Char RNN ( N:M RNN ������ �̿��� ) >

 -  apple �� �Է¹޾� pple!�� ����ϴ� RNN text generator ������ ���� RNN�� ���۹�� �˾ƺ��� 

    N:N (�ٴ��) RNN�� �̿��� �ؽ�Ʈ ���� ( without embedding layer )

    many to many 1�� ���� ( ��� ���� ó�� encoder, decoder ���� �� ���� �ƴ�, �� ���� �Է¿� ���� ����� ���� )

     O  0  0
     |  |  |
     �� �� ��
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

# �ܾ�����(vocabulary) ����
# {a, p, l ,e , !}
vocabulary = sorted(list(set(input_string + label)))

# integer embedding
char_embedding = { char: idx for idx, char in enumerate(vocabulary) }

# integer to word 
# training ���� , �н� ������ �˾ƺ��� ���� �����ϴ� �������� ���� 
number_to_word = {idx : char for idx, char in enumerate(vocabulary) }

# �Է� ������ ���� �Ӻ��� [1,4,4,3,2]
train_data = [ char_embedding[char] for char in input_string ]

# ��� ������ ���� �Ӻ��� [4,4,3,2,0]
label_data = [ char_embedding[char] for char in label ]

# RNN�� �Է��� batch�� ����� 3D Tensor �̸�,
# Tensor�� ���´� [ batch, row, col] �� ���¸� ���� ������ 
# �Է� ������ �� ���� �����͸� 3D tensor ȭ 
train_data, label_data = [train_data], [label_data]

# one hot vector ȭ 
# embedding layer�� ������� �ʱ� ������ ���� �Ӻ��� �� �����͸� 
# �ش� ������ �´� one-hot vector�� ��ȯ

# np.eye(size) : size ũ���� �밢 ��� ���� 
# ũ�Ⱑ 5�� �밢����� [1,0,0,0,0] , [0,1,0,0,0] ... [0,0,0,0,1] �� ���·� ���� �������Ƿ�
# train_data�� �� ���ڿ� ���� ���� �Ӻ��� �� ��ġ�� �밢����� �������� ������ one-hot vector�� �ǹ� 
one_hot_train_data = [ np.eye(len(vocabulary))[value] for value in train_data]

# vector�� tensorȭ 
x_train = torch.FloatTensor(one_hot_train_data)
y_test = torch.LongTensor(label_data)

# tensor dimension chk
# x_train = [1,5,5] , y_train = [1,5]
print(x_train.shape)
print(y_test.shape)



# Step 2 : Construct RNN model

# model params
# many to many first case�ν� �� ���� �Է¿� ���� ����� ���� ������
# �Է°� ����� ũ�Ⱑ ���ƾ� �Ѵ�
input_size = len(vocabulary)
hidden_size = 5
output_size = 5
learning_rate = 0.1

# RNN ����
#
#
#  (output area with FC)      output
#  ---------------------------  ��
#  (RNN cell area)
#                h_(t-1)  ->   h_(t)
#                               ��
#                              input

class Customized_RNN(torch.nn.Module):

    def __init__(self, input_size , hidden_size, output_size):

        super(Customized_RNN, self).__init__()

        # RNN cell ���� ( RNN cell area)
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

    # output.view(-1, input_size)  , batch ���� ���Ÿ� ���� ��� ( ��Ȯ�� ���� ��, batch_size��ŭ ������ ��� �����͸� ���ļ� ��Ȯ���� ��������� 3D -> 2D Tensor ȭ )
    # data.view(-1, size) : data�� ���� ���� ������ �����ϰ� [?, size] ���·� ��ȯ
    # example) If  data = [ 2,3,5] , size = 5 then  data.view[-1,5]  ->  data.shape = [6, 5] 
    loss = criterion( output.view(-1, input_size), y_test.view(-1) )
    
    # ���� ���
    loss.backward()

    # weight update
    optimizer.step()

    # ���� ��� �����ߴ��� �����ֱ� ���� code

    # ���� ������ �� �ش��ϴ� �� time_step �� 5���� vector�� ���� ���� ���� ���� index ���� 
    result = output.data.numpy().argmax(axis=2)
    
    # result.shape = (1,5) = [1,1,5] ���¶� np.squeeze()�� �̿��Ͽ� ���� �ٱ� ���� ����(���� ����)
    # ex) [[4 4 0 4 0 ]] => [4 4 0 4 0]
    result_string = ''.join( [number_to_word[value] for value in np.squeeze(result) ])  

    # loss value �� �����ҋ� loss.item()
    print("Epoch : ", epoch , "Loss : " , loss.item(), "Prediction : ", result, "Label : ", y_test, "Prediction string :", result_string)





