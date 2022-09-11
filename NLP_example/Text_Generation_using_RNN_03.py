# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< Word RNN ( N:M RNN 구조를 이용한 ) >

 - pytorch 의 embedding layer를 사용하여 단어 단위의 Text Generation 구성해보기


<Step>

1. Text Preprocessing

   1-1. 문장을 단어로 tokenization

   1-2. 단어에 대한 정수 임베딩 및 단어집합 형성 

   1-3. sentence =>  word embedding vector로 변환 및

        입력 & 정답 데이터 생성 

        입력 데이터 : sentence[:-1] , 정답 데이터 : sentence[1:]




2. Construct Model

   2-1. embedding layer가 추가 된 RNN 신경망 구성 

        raw_data -> embedding layer -> RNN_input -> hidden layer -> output layer


        * data의 차원 변화 

         if data.shape = [1,6] ( batch, time_step ) , after embedding layer then data.shape = [1,6,5] 
       
        why ? 
        
         =? 0, ... 5 에 해당하는 각 time step t에 대하여  t = [ 1,2, 3.3 , 4.2 ...] 와 같이 크기 5(사용자 정의) 만큼의 embedding vector로 변환 하기 때문 


        * RNN의 입력 
        
           x, _status= self.rnn(x) 의 형태를 보면 h_t를 사용하지 않는다

           => 이론적으로는 사용해야 하지만, default 값으로 셋팅 되있음

              기계번역 task 코딩에서는 명시적으로 h_t를 사용하는데, text 생성, 분류 파트에서는 그렇게 하지 않는 이유는 

              ->  

              ->

   2-2. construct model and set loss, optimizer

   2-3. Training


'''


import torch
import torch.nn as nn
import torch.optim as optim


sentence = "Repeat is the best medicine for memory"


# Step 1

# 1-1 : word tokenization
words = list(set(sentence.split()))

# 1-2 : 정수 임베딩 및 단어 집합 형성
vocab = { word : idx+1 for idx, word in enumerate(words) }
num2word = {idx+1 : word for idx, word in enumerate(words) }

# unknown token 에 대한 보정
vocab['<unk>'] = 0
num2word[0] = '<unk>'


# 1-3 : sentence를 word embedding vector로 변환 ( word 로 tokenization 하고, 해당 기준으로 문장을 정수 임베딩 화) 및 입력, 정답 데이터 형성

def make_data(sentence, vocab):

    sen_encoded = [vocab[word] for word in sentence.split()]

    _input, _label = sen_encoded[:-1] , sen_encoded[1:]

    # boath x_train.shape and label = [1,6] 
    # batch size가 없음으로 가장 바깥 차원에 1차원 추가
    # NN의 input은 batch_size를 고려한 3D tensor를 활용
    x_train = torch.LongTensor(_input).unsqueeze(0)
    label = torch.LongTensor(_label).unsqueeze(0)
    
    return x_train, label

# train_data.shape , label.shape = [1,6]
train_data, label_data = make_data(sentence, vocab)


# Step 2

# model hyperparams
init_dim, output_dim = len(vocab), len(vocab)

# RNN의 input dim은 embedding vector의 크기에 따라 결정 됨 
rnn_input_dim, hidden_dim = 5, 15

learning_rate = 0.1


# 2-1 : embedding layer가 포함된 RNN 구현

class JW_RNN(nn.Module):

    def __init__(self, init_dim, rnn_input_dim, hidden_dim, output_dim, batch_first = True ):

        super(JW_RNN, self).__init__()

        # raw data가 embedding layer를 통과하면, 사전에 정해진 차원 (rnn_input_dim) embedding vector로 변환
        self.embedding_layer = nn.Embedding(num_embeddings = init_dim, embedding_dim = rnn_input_dim)

        self.rnn = nn.RNN(rnn_input_dim, hidden_dim, batch_first = True )

        self.linear = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        # [1,6] -> [1,6,5]
        x = self.embedding_layer(x)
        
        # [1,6,5] -> [1,6,15]
        x, _status= self.rnn(x)
        
        # [1,6,15] -> [1,6,8]
        x  = self.linear(x)
        
        # 8개의 다음 단어 후보들과 비교해야 함으로 
        # 비교해야 하는 후보 차원을 제외한 나머지는 concatenate 하기 위한 -1 
        x = x.view(-1, len(vocab))

        return x 


# 2-2 : construct model and set loss & optimizer
net = JW_RNN(init_dim, rnn_input_dim, hidden_dim, output_dim)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(params = net.parameters())


# 2-3 : Training

for epoch in range(201):

    optimizer.zero_grad()

    output = net(train_data)

    loss = loss_function(output, label_data.view(-1))

    loss.backward()

    optimizer.step()


    prediction_string = ""

    # records

    if epoch % 40 == 0 :

        print("[{:02d}/201] {:.4f} ".format(epoch+1, loss))

        pred = output.softmax(-1).argmax(-1).tolist()

 
        prediction_string = "Repeat"

        for value in pred:

            prediction_string += " " + num2word[value]

        print(prediction_string)
        print()
