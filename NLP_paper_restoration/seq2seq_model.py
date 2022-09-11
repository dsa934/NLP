# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-03


 < seq2seq model setup >

  1. Encoder Layer
  
     * Many to one 

       => Encoder�� ��� context vector�� ����°��� ����������, �� time_step(t) ���� ��°��� ���� �ʿ䰡 �����Ƿ�,

          �Է� sequence data�� �ѹ��� ó��



     * Data_dim ��ȯ 

       1. encoder �Է� ������ =>  [ �ܾ� ��, batch_size ] = [21,128] 

          => batch_size : ����� ���� ���� (128) 

             �ܾ�� : 128���� seq input data �� ���� �� ����(���� �ܾ��� ���� ����) seq data �������� ����



       2. embedding layer ���� = [ �ܾ� ��, batch_size, embed_dim ] = [21, 128, 256]

          => embedding layer ���� �θ� ���� nn.Embedding(input_dim, embed_dim)���� ���� ���ִµ�, input_dim = len(src.vocab) = 7855 �̴�.
           
             ��, 1������ �ʱ� �Է� �����Ͱ� [21,128] �̶�� ���� 21���� �ܾ�� �̷���� sequence ������ 128�� �ִٴ� �ǹ� �̸�,

             1���� �ܾ�� �ٽ� input_dim ��ŭ�� one_hot vector�� ��Ÿ�� �� �ִ�.

             �׷��� one hot vector�� sparse vector(0�� ����) ������, �ڿ��� ȿ���� ����� ���� embed_dim���� ������ ��� �Ѵ�.

             ����, �� �ܾ�(word)�� ���� one-hot vector ȭ or ���� �Ӻ��� ����, embeddingd dim���� ���ν�Ű�� ������ nn.embeddings ���� ����  



       3. context vector = [ n_layer * bi_dir, batch_isze, hidden_dim ] = [2, 128, 512]

          => embedding vector�� ���� hidden state & cell state ���(LSTM, GRU Case)    
            
             bi_dir = 2 if bidirection = True else 1 
       



  2. Decoder Layer

     * Many to Many 

       => Decoder�� ��� �� time_step(t) ���� ��°��� �ʿ��ϱ� ������ �Է� sequence �ܾ �ϳ��� ó�� �ؾ� �Ѵ�.


     * Data_dim ��ȯ 

       1. decoder �Է� ������ =  [ batch_size ] = [128] 

          => every time_step ���� ��°��� ����ؾ� ������ �Է� �����ʹ� �ϳ��� ó�� 


       2. ���� �������� ���� unsqeeuze = [ �ܾ� ����, batch_size ] = [ 1, 128 ]

         => NN�� �Է��� ���� 2D or 3D tensor �Է����� �ޱ� ������ unsqueeze(0)�� ���� ������ ���� ���� 

          
       3. embedding layer = [ �ܾ� ��, batch_size, embed_dim] = [1,104,256]


       4. hidden & cell state and output ��� 
       
          output_dim = [1, 128, 512] ,  hidden & cell_dim = [2, 128, 512]

          => encoder�� context(hidden, cell) vector = [2, 128, 512] , decoder�� embedding vector = [1,128, 256] �� ��� 

         
       5. output_dim = [128, 5893] = [batch_size, len(trg.vocab)]

          => linear layer�� ���� Recurrent �Ű������ ���� ��� �� output value�� decoder�� input ������ �����ϰ� ��ȯ

             CrossEntropy�� ���� ���� �ܾ�� ������ ��� �ĺ���(len(trg.vocab)) ���� ��ȯ�Ͽ� ���� Ȯ���� ���� �ܾ �߷��� �� 

          




  cf : ������ ���� 

   * nn.RNN , nn.LSTM �� �ñ���

     => Q1. �̷����� ������ �� recurrent �迭 �Ű���� �Է����� ( hidden_state, input ) 2���� �ʿ�

            �׷��� ���� ������ ����, nn.rnn = nn.lstm(input_x) �� ���� ���·� hidden state�� ������ ��찡 ���

            why ? 

            => official doc�� �����ϸ�, hidden state, cell_state(lstm, gru case) �� �������� ���� ��� default ��(0) �ο�


        Q2. �ʱⰪ�� default�� �־����ٸ�, �� time_step �� hidden state�� �˾Ƽ� ������ �Ǵ� �� ?

           => official code�� ��� �� ��� �׷���. 

              ��, encdoer-decoder ���� ó�� Ư���� hidden state�� �Ѱ��ִ� ��찡 �ƴϸ�, hidden state ǥ��� ���� �Ǵ� ��찡 ���� 



   * ���� ���� ( teacher forcing , ���� ���� : t)

     => decoder ���� t-1�� output�� t������ �Է����� ����ϴ°��� �Ϲ������� < testing > �ܰ��϶� ����ϴ� ���

        < training > �ܰ迡���� t-1 ������ label ���� t ������ �Է����� ����Ѵ� 

        -> �Ʒð������� t-1 output�� ���� ��� t������ ������ �߸��� ���� ������, �̴� �����ۿ����� ���ڴ� ��ü ������ Ʋ���� �� �ִ�.


        but �ش� �ڵ忡���� train, test�� �Ϻ��ϰ� �и��ϴ� ���� �ƴ϶�, random ���̺귯���� ����

        ������ ���������ν� Ȯ�������� ���� t-1�� label �� or t-1�� output�� ����Ͽ� �н��� �Ѵٴ� �̾߱� 


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
        # seq2seq �� ���� 
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
        
        # prediction �� ���� Tensor ��ü ����
 
        # trg.shape = [128,28] = [batch, ] 
        target_length, batch_size = target.shape[0], target.shape[1]
                
        # target data�� �� �� �ִ� ��� �ܾ� �ĺ����� ���Ͽ� cross entropy�� ���� Ȯ���� ���������
        # decoder�� output�� 
        target_vocab_size = self.decoder.decoder_in_dim

        # predction_string = [25, 128, 5893] 
        prediction_string = torch.zeros(target_length, batch_size, target_vocab_size).to(self.device)

        # ���� ������ ���� <sos> ��ū���� ���� �ؾ���
        # ������ ���� ù������ �׻� <sos> ��ū 
        _input = target[0,:]

        for time_step in range(1, target_length):

            output, _hidden, _cell = self.decoder(_input, _hidden, _cell)

            # �� time_step���� ���� Ȯ���� ���� �ܾ �������忡 �߰�
            prediction_string[time_step] = output

            # output = [128,5893] ������ ���� Ȯ���� ���� �ܾ��� index ���� ���� 
            best_word = output.argmax(1)


            # teacher_forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # training step -> t-1 �� label ��(Target[time_step]) �̿�
            # testing step -> t-1�� output ��(best_word) �̿� 
            _input = target[time_step] if teacher_force else best_word

        return prediction_string







