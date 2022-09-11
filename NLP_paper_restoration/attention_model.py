# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-06



  1. Encoder 

     * Encoder is bidirectional GRU
  
     * Data dimension 변환 

       1. encoder input data = [ 단어의 수 , batch_size ] = [ 33, 128 ]

          => 각각의 iteration 마다 단어의 수는 다름 ( 해당 batch_size에서 가장 긴 문장을 기준으로 형성 with padding )


       2. embedding layer 이후, x_dim = [ 단어의 수, batch_size, embed_dim ] = [33, 128, 256]

          => x_dim이 갖는 의미 해석

             [1,1,256] : 33개의 단어 중 첫번째 단어, 128개의 sample 중 첫번째 sample의 차원이 256 

             embedding layer : nn.Linear(len(src.vocab), embed_dim) , 즉 7855 -> 256 변환 Linear Layer

             따라서, 1개의 단어가 갖는 one-hot vector dimension은 len(src.vocab) = 7855 이며, 이를 embedding layer를 통해 embed_dim 으로 word embedding 한 행위 

             즉, 128개의 sample을 구성하는 33개의 단어에 대해 한번에 word embedding 


       3. rnn layer 이후, output (for attention) , hidden(for context) dim 
       
          => output = [단어의 수, batch_size, n_layer * hidden_dim ] = [33, 128, 1024]

             hidden = [batch_size, hidden_dim] = [2, 128, 512]

             torch.nn.rnn 의 output dimension = [seq_len, batch_size, direction * hiddem_dim ] in official

             torch.nn.rnn 의 hidden dimension = [direction * n_layer , batch_size, hidden_dim] in official


       4. rnn layer 이후 hidden state dimension 보정
       
          => 현재 hidden state dimension = [2, 128,512] 이며, bidirectional GRU 임으로, 아래와 같이 구성되 있다.
             
             Forward hidden state = [1, 128, 512]   => 0 번쨰 idx : hidden[-2, :, :] 2D tensor

             Backward hidden state = [2, 128, 512]  => 1번쨰 idx  : hidden[-1, :, :] 2D tensor

             따라서 hidden state를 하나로 합치기 위하여 torch.cat( (hidden[-2,:,:], hidden[-1,:,:]), dim=1) 사용

             2D tensor 에 대하여 dim = 1 임으로, hidden_dim에 대하여 합쳐지기 떄문에 hidden_dim = [128,1024] 

             이후 FC layer(1024, 512) 와 torch.nn.tanh 를 통해 [128,512] 크기의 hidden_state로 변환 




  2. Attention 
  
     * Data dimension 변환 

       1. Attention input data ( 현재 시점 : t )

          => attention 사용을 위한 이론적 필수 요소 : [ 모든 time_step에 대한 encoder의 hidden state  = encoder_output, decoder의 t-1 시점의 hidden_state ] 

             if t=1 case (최초 시도):

                decoder_1 의 hidden_state를 계산하기 위해서는 [ decoder_0, encoder_output] 필요 

                decoder_0 = encoder's hidden에 해당 


            else : 

                최초 시도가 아닌 경우, 만약 t=5 라면 , decoder_5의 hidden state 계산을 위해 [ decoder_4, encoder_output ] 요구 

                즉, 이때의 hidden_state는 decoder에 의해 만들어 짐 


            따라서, decoder_output, hidden = self.decoder(seq2seq_input, hidden, encoder_output) 에서 알 수 있듯

            encoder의 경우, 각 time_step 별 hidden_state를 한번에 계산하는데 반해 ( encdoer code를 보면 nn.rnn(x) 이후에 새로운 hidden_state 넣는 행위가 없음 )

            decoder의 경우, 각 time_step 별 hidden_state가 초기에는 encoder에 의해 만들어진 hidden이지만, 이후부터는 decoder에 의해 생성


            ∴ 이미 구현 된torch.nn.rnn() 에 input_data 만 넣으면 hidden_state가 상속받은 nn.Module에 의해 내부에서 자체적으로 갱신되는 것으로 생각한 의문에 대한 해답



       2. Attention layers 

          => * nn.Attention( (encoder_hidden_dim *2 ) + decoder_hidden_dim , decoder_hidden_dim )

               - Encoder가 bidirectional GRU 임으로  forward, backward hidden state가 존재 하고, 이를 concatenation 했기 떄문에 encoder_hidden_dim * 2

                 현 시점 t를 기준으로, t-1 시점의 decoder hidden_state ( hidden ) 이 
            
                 encoder의 all time_step 별 hidden_state ( encdoer_output ) 각각에 대하여 얼마만큼 영향을 주는지 각각 표현하기 위해 hidden_dim (dim=2)을 기준으로 concatenation

                 따라서 attention layer의 입력 데이터 차원 =  (encoder_hidden_dim *2 ) + decoder_hidden_dim  이 되며 , 출력 차원은 = decoder_hidden_dim 


             * self.revise =  nn.Linear( decoder_hidden_dim, 1, bias = False) 
            
               - 각 단어에 해당하는 attention score value 값 변환 
               
                 [batch, 단어의 수, decoder_hidden_dim] -> [ batch, 단어의 수, 1 ] 로 변환 후, squeeze(2)를 통해 3D tensor -> 2D tensor 화 


             * softmax 

               - softmax fucntion을 통해 각 단어 별 energy score value를 확률 값으로 변환, 각 단어에 대한 가중치를 부여할 수 있음 

  


  3. Decoder 
  
     * Data dimension 변환 

       1. Decoder input data 
       
          => Decoder input , encoder_output, hidden

             decoder input의 경우  encoder의 입력(불어)과 다르게 영어 입력이 주어 지며(target) 

             decoder input_data dim = [128] , 각 time_step별로 한개씩 주어짐  

             input.unsqueeze(0) = [1,128] , embedding layer의 입력 데이터로 활용하기 위한 차원보정




       2. Calculate Attention in Decoder 

          * attention_vector = attention_vector.unsqueeze(1) = [batch, 1, 단어의 수] 

            -> attention 객체의 계산 결과 , attention_output = [batch, 단어의 수] 가 되므로, 1 개의 단어에 대한 각 단어들의 attention probability values 의미


          * encoder_output = encoder_output.permute(1,0,2)

            -> 원래 encoder_output_dim = [단어 수, batch_size, encoder_hidden_dim * direction]

               attention과의 행렬 곱(각 encoder output에 attention weight 부여)을 위한 차원 보정 

               encoder_output_dim.permute(1,0,2) = [batch_size, 단어 수, encdoer_hidden_dim * direction]


       3. Decoder GRU 

          * Decoder is not bidirectional GRU 

          * self.rnn = nn.GRU( (encoder_hidden * 2) + embed_dim, decoder_hidden_dim ) 

            -> * Decoder의 GRU's input data

                 a) decoder input을 embedding layer 통과시킨 embedding vector = [1, 128, 256]

                 b) attention vector = [128, 1, 1024]

                 위 두 요소의 concatenation 값을 사용함으로, 입력 차원은 (encoder_hidden * 2) + embed_dim


               * Decoder의 GRU's input hidden 

                 a) input hidden의 경우 = [128, 512] = [batch, hidden_dim] 임으로 

                    unsqueeze(0)을 통해 [1, batch, hidden_dim]  

                    즉, 1개의 단어에 대한 hidden 입력이라는 것을 2D Tensor 가 아닌 3D Tensor로써 명확하게 표현 & 차원 보정
             


               * Decoder GRU's output  

                 a) output = [단어 수, batch, decdoer_hidden_dim * direction] = [ 1, 128, 512]
             
                 b) hidden = [n_layer * direction, batch, decoder_hidden_dim ] = [ 1, 128, 512]


         

       4. Decoder output layers

          * output = [ 1, 128 ,512 ] 
          
          * embed_x = [1, 128, 256 ] 
          
          * 변환된 each_word_attention score = [1, 128, 1024]


          => 쓸모없는 0차원을 모두 제거 squeeze(0) 후,

             128개의 샘플 데이터에 대하여 decoder_input, input에 대한 attention score, decoder gru output을 연결 ( 128개의 샘플을 살려야 함으로, hidden_dim 축으로 합쳐야함 그래서 dim=1 )

             따라서,
             
             output layer의 입력차원  = ( encoder_hidden_dim *2 ) + decoder_input_dim + embedding_dim

             * decoder_input_dim : decoder는 입/출력 차원의 크기가 같음 

                                   기존에 RNN 구조 역시 매 time_step 마다 입력에 대한 출력을 내뱉을때 입출력 차원이 같아야만 하는 단점이 존재했는데,

                                   이것이 문제가 됬던 이유는 입력은 불어인데, 출력이 영어임으로 길이가 같지 않을 가능성이 높았기 떄문에 문제가 되었지만

                                   해당 구조에서는 encoder, decoder가 나뉘어져있으며, decoder의 입,출력이 모두 영어로 동일하기 떄문에 입,출력 차원이 같아도 상관 없다 



          

  cf : 구현시 참고 

   * assert ( 가정 설명문 )

     - 단순히 에러를 찾기 위함이 아닌, 값을 보증하기 위해 사용 
     
     - method 

       assert <조건> , <msg>

       list = [1,2,3,5 , 6.6 , 8]

       for value in list:

           assert type(value) is int, "정수가 아닌 값이 있네"


   * bmm

     - 3D Tensor 행렬 곱 연산

     - if a_tensor_dim = [b , n, m] and b_tensor_dim = [b, m, p] then result = bmm(a_tensor, b_tensor)

       result = [b , n, p] 임으로 곱하는 순서가 중요함 (차원을 계산하기 위해서는 )


     =>  each_word_attention_score = torch.bmm(attention_vector, encoder_output)

         attention_vector = [128, 1, 33 ]

         encoder_output = [33, 128, 1024] -> encoder_output.permute(1,0,2) = [128, 33, 1024]

         -> each_word_attention_score = [128, 1, 1024]

    

'''

import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, encoder_input, embed_dim, encoder_hidden, decoder_hidden, dropout_ratio):

        super().__init__()

         # dropout
        self.dropout = nn.Dropout(dropout_ratio)
       
        # embedding layers
        self.embedding = nn.Embedding(encoder_input, embed_dim)

        # bi-direction GRU
        self.rnn = nn.GRU(embed_dim, encoder_hidden, bidirectional = True)

        # Decoder의 hidden state 계산을 위해 양방향 RNN의 hidden state 차원 보정 
        self.fc = nn.Linear(encoder_hidden * 2, decoder_hidden )

     
    def forward(self, x):
        
        # init x_dim = [33, 128] = [ 단어의 수 , batch ]
        
        # embedding dim = [33, 128, 256] = [ 단어의 수, batch, embed_dim]
        embed_x = self.dropout( self.embedding(x) )
        
        # output = [33, 128, 1024] , hidden = [2, 128, 512] 
        encoder_output , encoder_hidden = self.rnn(embed_x)
        
        # bidirectional rnn의 hidden state 차원 보정
        # forward rnn 의 hidden : hidden[-2, :, : ]
        # backward rnn 의 hidden : hidden[-1, :, :] 
        # bidirectional_hidden_state = concate (foward_hidden, backward_hidden)
        encoder_hidden =  torch.cat( (encoder_hidden[-2,:,:,] , encoder_hidden[-1,:,:] ), dim = 1) 
        
        encoder_hidden = torch.tanh( self.fc(encoder_hidden) )

        return encoder_output, encoder_hidden



class Attention(nn.Module):

    def __init__(self, encoder_hidden, decoder_hidden):
        
        super().__init__()
        
        # 자세한 설명은 Attention input data  참조
        self.attention = nn.Linear((encoder_hidden * 2) + decoder_hidden, decoder_hidden )

        self.revise = nn.Linear( decoder_hidden, 1, bias = False )

    def forward(self, encoder_output, hidden):
        
        # encoder_output = [33, 128, 1024] , hidden = [128, 512] 
        word_len = encoder_output.shape[0]

        # hidden_dim = [ 128, 33, 512 ] 
        # repeat() : 
        hidden = hidden.unsqueeze(1).repeat(1, word_len, 1)

        # hidden , encoder_output을 concatenation 하기 위한 차원 보정 
        # encoder_output = [128, 33, 1024]
        encoder_output = encoder_output.permute(1,0,2)
        
        attention_value = self.attention( torch.cat( (hidden, encoder_output), dim = 2) )

        # attention_energy = [128, 33, 512]
        attention_energy = torch.tanh( attention_value)

        # 각 단어 당 attention energy value로 만들기 위해 revise layer를 통한 차원 변환 (512 -> 1 )
        # attention_weight = [128,33]
        attention_weight = torch.nn.functional.softmax( self.revise(attention_energy).squeeze(2) , dim = 1)
        
        return attention_weight




class Decoder(nn.Module):

    def __init__(self, decoder_input_dim, embed_dim, encoder_hidden, decoder_hidden, dropout_ratio, attention):
        
        super().__init__()

        self.decoder_input_dim = decoder_input_dim
        self.attention = attention

        # dropout layers
        self.dropout = nn.Dropout(dropout_ratio)

        # embedding layers
        self.embedding = nn.Embedding(decoder_input_dim, embed_dim)

        # decoder GRU , 입력값으로 attention & decoder input 취급
        self.rnn = nn.GRU( (encoder_hidden * 2) + embed_dim, decoder_hidden )

        # output layer
        self.output = nn.Linear( (encoder_hidden *2 ) + embed_dim + decoder_hidden  , decoder_input_dim)


    def forward(self, x, hidden, encoder_output):
        
        # x = [1, 128] , output = [24, 128, 1024] , hidden = [128, 512]
        x = x.unsqueeze(0)

        # embedding x = [1, 128, 256] 
        embed_x = self.dropout( self.embedding(x) )
 
        # attention_vector = [128,33] 
        attention_vector = self.attention( encoder_output, hidden ) 
   
        attention_vector = attention_vector.unsqueeze(1)

        # encoder output 을 attention 차원계산이 가능하도록 보정
        encoder_output = encoder_output.permute(1,0,2)

        # encoder의 입력 단어에 대하여 attention 가중치 부여 
        # each_word_attention_score = [128,1,1024] 
        # bmm 참조 
        each_word_attention_score = torch.bmm(attention_vector, encoder_output)
    
        # rnn 계산을 위한 차원 변환
        each_word_attention_score = each_word_attention_score.permute(1,0,2)

        # 각 decoder 입력에 대하여 attention으로 계산된 가중치 score를 concatenate
        x = torch.cat( (embed_x, each_word_attention_score), dim = 2)

        # decoder's hidden state 
        # decoder_output = [1,104, 512], hidden = [1, 104, 512] 
        decoder_output, hidden = self.rnn(x, hidden.unsqueeze(0) )
 
        # assert (가정 설명문)
        # decoder에서는 단어의 수 ( 1개의 time_step씩 계산) , n_layer, direction 모두 1의 값을 갖기 떄문에
        # output, hidden의 값이 동일

        assert (decoder_output == hidden).all()

        embed_x = embed_x.squeeze(0)

        decoder_output = decoder_output.squeeze(0)

        each_word_attention_score = each_word_attention_score.squeeze(0)
        


        _pred = self.output( torch.cat( (decoder_output, embed_x, each_word_attention_score), dim = 1) )

        return _pred, hidden.squeeze(0)


import random 


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device


    def forward(self, x, target, teacher_forcing_ratio = 0.5 ):

        # encoder hidden이 아닌 hidden이라고 표기한 이유는 Process of Seq2Seq hidden dim 참조
        encoder_output, hidden = self.encoder(x)
        
        # 예측 문장을 기록하기 위한 result tensor의 형태 구축
        # result tensor = []
        target_len, batch_size, target_vocab_size = target.shape[0], target.shape[1], self.decoder.decoder_input_dim

        seq2seq_output = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        seq2seq_input = target[0,:]

        for time_step in range(1, target_len):

            decoder_output, hidden = self.decoder(seq2seq_input, hidden, encoder_output)

            # 
            seq2seq_output[time_step] = decoder_output

            # output = [128,5893] 임으로 가장 확률이 높은 단어의 index 정보 저장 
            best_word = decoder_output.argmax(1)

            # teacher_forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # training step -> t-1 의 label 값(Target[time_step]) 이용
            # testing step -> t-1의 output 값(best_word) 이용 
            seq2seq_input = target[time_step] if teacher_force else best_word
            

        return seq2seq_output