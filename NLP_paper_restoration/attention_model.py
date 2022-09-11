# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-06



  1. Encoder 

     * Encoder is bidirectional GRU
  
     * Data dimension ��ȯ 

       1. encoder input data = [ �ܾ��� �� , batch_size ] = [ 33, 128 ]

          => ������ iteration ���� �ܾ��� ���� �ٸ� ( �ش� batch_size���� ���� �� ������ �������� ���� with padding )


       2. embedding layer ����, x_dim = [ �ܾ��� ��, batch_size, embed_dim ] = [33, 128, 256]

          => x_dim�� ���� �ǹ� �ؼ�

             [1,1,256] : 33���� �ܾ� �� ù��° �ܾ�, 128���� sample �� ù��° sample�� ������ 256 

             embedding layer : nn.Linear(len(src.vocab), embed_dim) , �� 7855 -> 256 ��ȯ Linear Layer

             ����, 1���� �ܾ ���� one-hot vector dimension�� len(src.vocab) = 7855 �̸�, �̸� embedding layer�� ���� embed_dim ���� word embedding �� ���� 

             ��, 128���� sample�� �����ϴ� 33���� �ܾ ���� �ѹ��� word embedding 


       3. rnn layer ����, output (for attention) , hidden(for context) dim 
       
          => output = [�ܾ��� ��, batch_size, n_layer * hidden_dim ] = [33, 128, 1024]

             hidden = [batch_size, hidden_dim] = [2, 128, 512]

             torch.nn.rnn �� output dimension = [seq_len, batch_size, direction * hiddem_dim ] in official

             torch.nn.rnn �� hidden dimension = [direction * n_layer , batch_size, hidden_dim] in official


       4. rnn layer ���� hidden state dimension ����
       
          => ���� hidden state dimension = [2, 128,512] �̸�, bidirectional GRU ������, �Ʒ��� ���� ������ �ִ�.
             
             Forward hidden state = [1, 128, 512]   => 0 ���� idx : hidden[-2, :, :] 2D tensor

             Backward hidden state = [2, 128, 512]  => 1���� idx  : hidden[-1, :, :] 2D tensor

             ���� hidden state�� �ϳ��� ��ġ�� ���Ͽ� torch.cat( (hidden[-2,:,:], hidden[-1,:,:]), dim=1) ���

             2D tensor �� ���Ͽ� dim = 1 ������, hidden_dim�� ���Ͽ� �������� ������ hidden_dim = [128,1024] 

             ���� FC layer(1024, 512) �� torch.nn.tanh �� ���� [128,512] ũ���� hidden_state�� ��ȯ 




  2. Attention 
  
     * Data dimension ��ȯ 

       1. Attention input data ( ���� ���� : t )

          => attention ����� ���� �̷��� �ʼ� ��� : [ ��� time_step�� ���� encoder�� hidden state  = encoder_output, decoder�� t-1 ������ hidden_state ] 

             if t=1 case (���� �õ�):

                decoder_1 �� hidden_state�� ����ϱ� ���ؼ��� [ decoder_0, encoder_output] �ʿ� 

                decoder_0 = encoder's hidden�� �ش� 


            else : 

                ���� �õ��� �ƴ� ���, ���� t=5 ��� , decoder_5�� hidden state ����� ���� [ decoder_4, encoder_output ] �䱸 

                ��, �̶��� hidden_state�� decoder�� ���� ����� �� 


            ����, decoder_output, hidden = self.decoder(seq2seq_input, hidden, encoder_output) ���� �� �� �ֵ�

            encoder�� ���, �� time_step �� hidden_state�� �ѹ��� ����ϴµ� ���� ( encdoer code�� ���� nn.rnn(x) ���Ŀ� ���ο� hidden_state �ִ� ������ ���� )

            decoder�� ���, �� time_step �� hidden_state�� �ʱ⿡�� encoder�� ���� ������� hidden������, ���ĺ��ʹ� decoder�� ���� ����


            �� �̹� ���� ��torch.nn.rnn() �� input_data �� ������ hidden_state�� ��ӹ��� nn.Module�� ���� ���ο��� ��ü������ ���ŵǴ� ������ ������ �ǹ��� ���� �ش�



       2. Attention layers 

          => * nn.Attention( (encoder_hidden_dim *2 ) + decoder_hidden_dim , decoder_hidden_dim )

               - Encoder�� bidirectional GRU ������  forward, backward hidden state�� ���� �ϰ�, �̸� concatenation �߱� ������ encoder_hidden_dim * 2

                 �� ���� t�� ��������, t-1 ������ decoder hidden_state ( hidden ) �� 
            
                 encoder�� all time_step �� hidden_state ( encdoer_output ) ������ ���Ͽ� �󸶸�ŭ ������ �ִ��� ���� ǥ���ϱ� ���� hidden_dim (dim=2)�� �������� concatenation

                 ���� attention layer�� �Է� ������ ���� =  (encoder_hidden_dim *2 ) + decoder_hidden_dim  �� �Ǹ� , ��� ������ = decoder_hidden_dim 


             * self.revise =  nn.Linear( decoder_hidden_dim, 1, bias = False) 
            
               - �� �ܾ �ش��ϴ� attention score value �� ��ȯ 
               
                 [batch, �ܾ��� ��, decoder_hidden_dim] -> [ batch, �ܾ��� ��, 1 ] �� ��ȯ ��, squeeze(2)�� ���� 3D tensor -> 2D tensor ȭ 


             * softmax 

               - softmax fucntion�� ���� �� �ܾ� �� energy score value�� Ȯ�� ������ ��ȯ, �� �ܾ ���� ����ġ�� �ο��� �� ���� 

  


  3. Decoder 
  
     * Data dimension ��ȯ 

       1. Decoder input data 
       
          => Decoder input , encoder_output, hidden

             decoder input�� ���  encoder�� �Է�(�Ҿ�)�� �ٸ��� ���� �Է��� �־� ����(target) 

             decoder input_data dim = [128] , �� time_step���� �Ѱ��� �־���  

             input.unsqueeze(0) = [1,128] , embedding layer�� �Է� �����ͷ� Ȱ���ϱ� ���� ��������




       2. Calculate Attention in Decoder 

          * attention_vector = attention_vector.unsqueeze(1) = [batch, 1, �ܾ��� ��] 

            -> attention ��ü�� ��� ��� , attention_output = [batch, �ܾ��� ��] �� �ǹǷ�, 1 ���� �ܾ ���� �� �ܾ���� attention probability values �ǹ�


          * encoder_output = encoder_output.permute(1,0,2)

            -> ���� encoder_output_dim = [�ܾ� ��, batch_size, encoder_hidden_dim * direction]

               attention���� ��� ��(�� encoder output�� attention weight �ο�)�� ���� ���� ���� 

               encoder_output_dim.permute(1,0,2) = [batch_size, �ܾ� ��, encdoer_hidden_dim * direction]


       3. Decoder GRU 

          * Decoder is not bidirectional GRU 

          * self.rnn = nn.GRU( (encoder_hidden * 2) + embed_dim, decoder_hidden_dim ) 

            -> * Decoder�� GRU's input data

                 a) decoder input�� embedding layer �����Ų embedding vector = [1, 128, 256]

                 b) attention vector = [128, 1, 1024]

                 �� �� ����� concatenation ���� ���������, �Է� ������ (encoder_hidden * 2) + embed_dim


               * Decoder�� GRU's input hidden 

                 a) input hidden�� ��� = [128, 512] = [batch, hidden_dim] ������ 

                    unsqueeze(0)�� ���� [1, batch, hidden_dim]  

                    ��, 1���� �ܾ ���� hidden �Է��̶�� ���� 2D Tensor �� �ƴ� 3D Tensor�ν� ��Ȯ�ϰ� ǥ�� & ���� ����
             


               * Decoder GRU's output  

                 a) output = [�ܾ� ��, batch, decdoer_hidden_dim * direction] = [ 1, 128, 512]
             
                 b) hidden = [n_layer * direction, batch, decoder_hidden_dim ] = [ 1, 128, 512]


         

       4. Decoder output layers

          * output = [ 1, 128 ,512 ] 
          
          * embed_x = [1, 128, 256 ] 
          
          * ��ȯ�� each_word_attention score = [1, 128, 1024]


          => ������� 0������ ��� ���� squeeze(0) ��,

             128���� ���� �����Ϳ� ���Ͽ� decoder_input, input�� ���� attention score, decoder gru output�� ���� ( 128���� ������ ����� ������, hidden_dim ������ ���ľ��� �׷��� dim=1 )

             ����,
             
             output layer�� �Է�����  = ( encoder_hidden_dim *2 ) + decoder_input_dim + embedding_dim

             * decoder_input_dim : decoder�� ��/��� ������ ũ�Ⱑ ���� 

                                   ������ RNN ���� ���� �� time_step ���� �Է¿� ���� ����� �������� ����� ������ ���ƾ߸� �ϴ� ������ �����ߴµ�,

                                   �̰��� ������ ��� ������ �Է��� �Ҿ��ε�, ����� ���������� ���̰� ���� ���� ���ɼ��� ���ұ� ������ ������ �Ǿ�����

                                   �ش� ���������� encoder, decoder�� ��������������, decoder�� ��,����� ��� ����� �����ϱ� ������ ��,��� ������ ���Ƶ� ��� ���� 



          

  cf : ������ ���� 

   * assert ( ���� ���� )

     - �ܼ��� ������ ã�� ������ �ƴ�, ���� �����ϱ� ���� ��� 
     
     - method 

       assert <����> , <msg>

       list = [1,2,3,5 , 6.6 , 8]

       for value in list:

           assert type(value) is int, "������ �ƴ� ���� �ֳ�"


   * bmm

     - 3D Tensor ��� �� ����

     - if a_tensor_dim = [b , n, m] and b_tensor_dim = [b, m, p] then result = bmm(a_tensor, b_tensor)

       result = [b , n, p] ������ ���ϴ� ������ �߿��� (������ ����ϱ� ���ؼ��� )


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

        # Decoder�� hidden state ����� ���� ����� RNN�� hidden state ���� ���� 
        self.fc = nn.Linear(encoder_hidden * 2, decoder_hidden )

     
    def forward(self, x):
        
        # init x_dim = [33, 128] = [ �ܾ��� �� , batch ]
        
        # embedding dim = [33, 128, 256] = [ �ܾ��� ��, batch, embed_dim]
        embed_x = self.dropout( self.embedding(x) )
        
        # output = [33, 128, 1024] , hidden = [2, 128, 512] 
        encoder_output , encoder_hidden = self.rnn(embed_x)
        
        # bidirectional rnn�� hidden state ���� ����
        # forward rnn �� hidden : hidden[-2, :, : ]
        # backward rnn �� hidden : hidden[-1, :, :] 
        # bidirectional_hidden_state = concate (foward_hidden, backward_hidden)
        encoder_hidden =  torch.cat( (encoder_hidden[-2,:,:,] , encoder_hidden[-1,:,:] ), dim = 1) 
        
        encoder_hidden = torch.tanh( self.fc(encoder_hidden) )

        return encoder_output, encoder_hidden



class Attention(nn.Module):

    def __init__(self, encoder_hidden, decoder_hidden):
        
        super().__init__()
        
        # �ڼ��� ������ Attention input data  ����
        self.attention = nn.Linear((encoder_hidden * 2) + decoder_hidden, decoder_hidden )

        self.revise = nn.Linear( decoder_hidden, 1, bias = False )

    def forward(self, encoder_output, hidden):
        
        # encoder_output = [33, 128, 1024] , hidden = [128, 512] 
        word_len = encoder_output.shape[0]

        # hidden_dim = [ 128, 33, 512 ] 
        # repeat() : 
        hidden = hidden.unsqueeze(1).repeat(1, word_len, 1)

        # hidden , encoder_output�� concatenation �ϱ� ���� ���� ���� 
        # encoder_output = [128, 33, 1024]
        encoder_output = encoder_output.permute(1,0,2)
        
        attention_value = self.attention( torch.cat( (hidden, encoder_output), dim = 2) )

        # attention_energy = [128, 33, 512]
        attention_energy = torch.tanh( attention_value)

        # �� �ܾ� �� attention energy value�� ����� ���� revise layer�� ���� ���� ��ȯ (512 -> 1 )
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

        # decoder GRU , �Է°����� attention & decoder input ���
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

        # encoder output �� attention ��������� �����ϵ��� ����
        encoder_output = encoder_output.permute(1,0,2)

        # encoder�� �Է� �ܾ ���Ͽ� attention ����ġ �ο� 
        # each_word_attention_score = [128,1,1024] 
        # bmm ���� 
        each_word_attention_score = torch.bmm(attention_vector, encoder_output)
    
        # rnn ����� ���� ���� ��ȯ
        each_word_attention_score = each_word_attention_score.permute(1,0,2)

        # �� decoder �Է¿� ���Ͽ� attention���� ���� ����ġ score�� concatenate
        x = torch.cat( (embed_x, each_word_attention_score), dim = 2)

        # decoder's hidden state 
        # decoder_output = [1,104, 512], hidden = [1, 104, 512] 
        decoder_output, hidden = self.rnn(x, hidden.unsqueeze(0) )
 
        # assert (���� ����)
        # decoder������ �ܾ��� �� ( 1���� time_step�� ���) , n_layer, direction ��� 1�� ���� ���� ������
        # output, hidden�� ���� ����

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

        # encoder hidden�� �ƴ� hidden�̶�� ǥ���� ������ Process of Seq2Seq hidden dim ����
        encoder_output, hidden = self.encoder(x)
        
        # ���� ������ ����ϱ� ���� result tensor�� ���� ����
        # result tensor = []
        target_len, batch_size, target_vocab_size = target.shape[0], target.shape[1], self.decoder.decoder_input_dim

        seq2seq_output = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        seq2seq_input = target[0,:]

        for time_step in range(1, target_len):

            decoder_output, hidden = self.decoder(seq2seq_input, hidden, encoder_output)

            # 
            seq2seq_output[time_step] = decoder_output

            # output = [128,5893] ������ ���� Ȯ���� ���� �ܾ��� index ���� ���� 
            best_word = decoder_output.argmax(1)

            # teacher_forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # training step -> t-1 �� label ��(Target[time_step]) �̿�
            # testing step -> t-1�� output ��(best_word) �̿� 
            seq2seq_input = target[time_step] if teacher_force else best_word
            

        return seq2seq_output