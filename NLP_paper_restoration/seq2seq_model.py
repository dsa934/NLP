# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-03


 < seq2seq model setup >

  1. Encoder Layer
  
     * Many to one 

       => Encoder의 경우 context vector를 만드는것이 목적임으로, 매 time_step(t) 마다 출력값을 뱉을 필요가 없으므로,

          입력 sequence data를 한번에 처리



     * Data_dim 변환 

       1. encoder 입력 데이터 =>  [ 단어 수, batch_size ] = [21,128] 

          => batch_size : 사용자 임의 설정 (128) 

             단어개수 : 128개의 seq input data 중 가장 긴 문장(가장 단어의 수가 많은) seq data 기준으로 설정



       2. embedding layer 적용 = [ 단어 수, batch_size, embed_dim ] = [21, 128, 256]

          => embedding layer 선언 부를 보면 nn.Embedding(input_dim, embed_dim)으로 구성 되있는데, input_dim = len(src.vocab) = 7855 이다.
           
             즉, 1번에서 초기 입력 데이터가 [21,128] 이라는 것은 21개의 단어로 이루어진 sequence 문장이 128개 있다는 의미 이며,

             1개의 단어는 다시 input_dim 만큼의 one_hot vector로 나타낼 수 있다.

             그러나 one hot vector는 sparse vector(0이 많음) 임으로, 자원의 효율적 사용을 위해 embed_dim으로 차원을 축소 한다.

             따라서, 각 단어(word)에 대한 one-hot vector 화 or 정수 임베딩 이후, embeddingd dim으로 매핑시키는 역할을 nn.embeddings 에서 진행  



       3. context vector = [ n_layer * bi_dir, batch_isze, hidden_dim ] = [2, 128, 512]

          => embedding vector를 통해 hidden state & cell state 계산(LSTM, GRU Case)    
            
             bi_dir = 2 if bidirection = True else 1 
       



  2. Decoder Layer

     * Many to Many 

       => Decoder의 경우 매 time_step(t) 마다 출력값이 필요하기 떄문에 입력 sequence 단어를 하나씩 처리 해야 한다.


     * Data_dim 변환 

       1. decoder 입력 데이터 =  [ batch_size ] = [128] 

          => every time_step 마다 출력값을 계산해야 함으로 입력 데이터는 하나씩 처리 


       2. 차원 형보정을 위한 unsqeeuze = [ 단어 개수, batch_size ] = [ 1, 128 ]

         => NN의 입력은 보통 2D or 3D tensor 입력으로 받기 떄문에 unsqueeze(0)을 통해 차원의 형태 보정 

          
       3. embedding layer = [ 단어 수, batch_size, embed_dim] = [1,104,256]


       4. hidden & cell state and output 계산 
       
          output_dim = [1, 128, 512] ,  hidden & cell_dim = [2, 128, 512]

          => encoder의 context(hidden, cell) vector = [2, 128, 512] , decoder의 embedding vector = [1,128, 256] 로 계산 

         
       5. output_dim = [128, 5893] = [batch_size, len(trg.vocab)]

          => linear layer를 통해 Recurrent 신경망으로 부터 계산 된 output value를 decoder의 input 차원과 동일하게 변환

             CrossEntropy를 통해 다음 단어로 가능한 모든 후보군(len(trg.vocab)) 으로 변환하여 가장 확률이 높은 단어를 추려야 함 

          




  cf : 구현시 참고 

   * nn.RNN , nn.LSTM 의 궁굼증

     => Q1. 이론으로 공부할 떄 recurrent 계열 신경망은 입력으로 ( hidden_state, input ) 2개가 필요

            그런데 실제 구현에 들어가면, nn.rnn = nn.lstm(input_x) 와 같은 형태로 hidden state가 생략된 경우가 빈번

            why ? 

            => official doc을 참조하면, hidden state, cell_state(lstm, gru case) 가 제공되지 않을 경우 default 값(0) 부여


        Q2. 초기값이 default로 주어진다면, 각 time_step 별 hidden state는 알아서 적용이 되는 것 ?

           => official code를 뜯어 본 결과 그렇다. 

              즉, encdoer-decoder 구조 처럼 특정한 hidden state를 넘겨주는 경우가 아니면, hidden state 표기는 생략 되는 경우가 많음 



   * 교사 강요 ( teacher forcing , 현재 시점 : t)

     => decoder 에서 t-1의 output을 t시점의 입력으로 사용하는것은 일반적으로 < testing > 단계일때 사용하는 방법

        < training > 단계에서는 t-1 시점의 label 값을 t 시점의 입력으로 사용한다 

        -> 훈련과정에서 t-1 output을 넣을 경우 t시점의 예측이 잘못될 수도 있으며, 이는 연쇄작용으로 디코더 전체 예측이 틀어질 수 있다.


        but 해당 코드에서는 train, test를 완벽하게 분리하는 것이 아니라, random 라이브러리를 통해

        난수를 생성함으로써 확률적으로 실제 t-1의 label 값 or t-1의 output을 사용하여 학습을 한다는 이야기 


'''

import torch
import torch.nn as nn
import random

class Encoder(nn.Module):

    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):

        super().__init__()
        
        # embedding layers 
        self.embedding = nn.Embedding(input_dim, embed_dim)

        
        # LSTM
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout = dropout_ratio )
        
        # dropout
        self.dropout = nn.Dropout(dropout_ratio)
        

    def forward(self, x):

        # init x_dim = [21,128]
        #print("encoder init input shape : ", x.shape)
        
        # embedding x_dim = [21, 128, 256] 
        embedding_x = self.dropout( self.embedding(x) ) 
        #print("encoder embedding:",embedding_x.shape)

        # outputs = [],  hidden = [2, 128, 512] , cell = [2,128, 512] 
        outputs, (hidden, cell) = self.rnn(embedding_x)
        #print("encoder hiddne, cell" , hidden.shape , cell.shape)

        return hidden, cell



class Decoder(nn.Module):

    def __init__(self, decoder_in_dim, embed_dim, hidden_dim, n_layers, dropout_ratio):

        super().__init__()

        # embedding layers
        self.embedding = nn.Embedding(decoder_in_dim, embed_dim)

        # LSTM
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout = dropout_ratio)

        # output layers
        # seq2seq 의 결함 
        self.decoder_in_dim = decoder_in_dim
        self.fc = nn.Linear(hidden_dim, decoder_in_dim)

        # dropdout
        self.dropout = nn.Dropout(dropout_ratio)


    def forward(self, x, h_status, cell):

        # init x_dim = [128]
        #print("decoder init input shape : ", x.shape)
        x = x.unsqueeze(0)

        # unsqueeze x_dim = [1,128]
        #print("unsqeeuze:", x.shape)

        # embedding x_dim = [1, 128, 256] 
        embedding_x = self.dropout( self.embedding(x) ) 
        #print("after decoder embedding",embedding_x.shape)

        # output_dim= [1, 128, 512], hidden & cell_dim = [2, 128, 512]
        output, (h_status, cell) = self.rnn(embedding_x, (h_status, cell)  )
        #print("After decoding output, h_status, cell", output.shape, h_status.shape, cell.shape)
        
        # _pred dim = [128,5893]
        _pred = self.fc(output.squeeze(0))

        #print("predict shape", _pred.shape)

        return _pred, h_status, cell


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, x, target, teacher_forcing_ratio = 0.5 ):

        #print("seq2seq start")
        #print("seq2seq init x shape :", x.shape)
        #print("seq2seq init target shape : ", target.shape)
        _hidden, _cell = self.encoder(x)
        
        # prediction 에 대한 Tensor 객체 형성
 
        # trg.shape = [128,28] = [batch, ] 
        target_length, batch_size = target.shape[0], target.shape[1]
                
        # target data가 될 수 있는 모든 단어 후보군에 대하여 cross entropy를 통해 확률을 계산함으로
        # decoder의 output은 
        target_vocab_size = self.decoder.decoder_in_dim

        # predction_string = [25, 128, 5893] 
        prediction_string = torch.zeros(target_length, batch_size, target_vocab_size).to(self.device)

        # 번역 시작을 위해 <sos> 토큰으로 시작 해야함
        # 문장의 가장 첫번쨰는 항상 <sos> 토큰 
        _input = target[0,:]

        for time_step in range(1, target_length):

            output, _hidden, _cell = self.decoder(_input, _hidden, _cell)

            # 현 time_step에서 가장 확률이 높은 단어를 예측문장에 추가
            prediction_string[time_step] = output

            # output = [128,5893] 임으로 가장 확률이 높은 단어의 index 정보 저장 
            best_word = output.argmax(1)


            # teacher_forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # training step -> t-1 의 label 값(Target[time_step]) 이용
            # testing step -> t-1의 output 값(best_word) 이용 
            _input = target[time_step] if teacher_force else best_word

        return prediction_string







