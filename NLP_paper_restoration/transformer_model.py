# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-15


< Transformer Model's dimension chnage Analysis >

 - 입력 데이터가 transformer model을 구축하는 각 class를 통과 하면서 변화하는 차원을 중점적으로 서술

 - Transforemr -> Encoder -> Encoder layers -> Multi-head attention -> decoder -> decoder_layers -> Multi-head attention



 < Transformer >

 * transformer input 

   - train / label data
    
     train = [128,30]  = [batch, train_seq_len]  ,label = [128,27]  = [batch, label_seq_len] 
  
     => label의 <eos> token을 제외하고 transformer의 입력 데이터로 활용 됨으로, 실제 label_seq_len -1 
     
 * train, label mask

   -  train_mask = [128,1,1,30] = [batch_size, 1, 1, train_seq_len] => 128개의 샘플, 1개의 head , 1개의 query 단어에 대한 30개의 key 값에 대한 mask

      label_mask = [128,1,1,26] = [batch_size, 1, 1, (label_seq_len -1 ) ]



 * masked train / label data 사용


 < Encoder >

 * encoder input 

   - word embedding vector : raw_train_data = [128,30] -> embedding vector = [128, 30, 256]

     pos embedding vector  : raw_pos_data = [128,30] -> embedding vector = [ 128, 30, 256 ]

     enc_scale_factor      : word embedding vector에 차별성을 부여하기 위해 사용 

     
 < Encoder_layer >

 * Encoder_layer input 

   - encoder's 결과물 word_embedding_vector * encoder_scale_factor ) + pos_embedding_vector = [128,30,256]


 * flow 

   - init -> Encdoer self Attention -> Residual add & Norm -> simple ff layers -> Residual add & Norm

     [128, 30, 256] -> [128,30,256] -> [128,30,256] -> [128,30,256] -> [128,30,256] 



 < Multi - head Attention ★ >

 * Encoder self attention 
 
   1) encoder self atten input dim    : 3 train_embedding_vectors = [128,30,256] x 3 , train_mask = [128, 1, 1, 30]

   2) Q, k, V vector 만들기           : Query / Key / Value vectors = [128, 30, 256]

   3) n개의 head로 vector 쌍 분할     : query/key/value. veiw(batch, -1, n_head, head_dim) . permute(0,2,1,3)

                                        [128,30,256] -> [128, 30, 8, 32] -> [ 128, 8, 30, 32] = [ batch, n_head, train_seq_len, head_dim]
    
   4) Attention Energy(AE)            : torch.matmul(Query, Key) : 3차원 이상 tensor 행렬곱 [b,n a,b] * [b, n, b, c] = [b, n, a, c]
   
                                        torch.matmul(query, Key.permute(0,1,3,2))

                                        [128,8,30,32] * [ 128, 8, 32, 30 ] = [128, 8, 30, 30] 

   5) softmax(AE dim = -1 )           : AE 마지막 차원 = train_seq_len , 각 Query에 대한 key 값들의 attention probability vbvalue 의미

                                        attention score = [128,8,30,30]

   6) word info                       : 각 단어에 대한 확률값 * 각 단어의 실제값 = 각 단어가 갖는 word information

                                        [128, 8, 30, 30] * [128, 8, 30, 32] = [128, 8, 30, 32]

   7) Concatenation                   : 1개의 encoder/decoder layer는 h개의 head를 갖고, 위 연산은 각 head 들에 대한 병렬 연산 임으로

                                        multi-head의 head를 concatenation 함으로써 입력 차원의 크기와 동일하게 조정

                                        [128,8,30,32] -> [128,30,8,32] -> [ 128,30, 256] 

   8) output layers                   : [ 128,30,256 ] ->  [128,30,256]



 * Masked Decoder Self Attention 

   [128,30,256] -> [128,30,256] -> [128.8,27,32] -> [128,8,27,27] -> [128,8,27,32] -> [128,27,256] -> [128,27,256]  
 

 * Encoder - Decoder  attention 

   1) Encoder decoder atten dim       : label = [128,27,256] , enc_output = [128,30,256], enc_output = [128,30,256]

   2) Q, k, V vector 만들기           : Query = [128,27,256] , Key = [128,30,256]  Value = [128,30,256]

   3) n개의 head로 vector 쌍 분할     : query/key/value. veiw(batch, -1, n_head, head_dim) . permute(0,2,1,3)

                                        Query : [ batch, n_head, label_seq_len, head_dim] = [128,8,27,32] 

                                        K & V : [ batch, n_head, train_seq_len, head_dim] = [128,8,30,32] 

   4) Attention Energy(AE)            : torch.matmul(query, Key.permute(0,1,3,2))

                                        query = [128, 8, 27, 32], key = [128, 8, 32, 30]

                                        [128,8,27,32] * [ 128, 8, 32, 30 ] = [128, 8, 27, 30] 

                                      * Encoder-decoder Attention에서 encoder_mask가 사용 되는 이유 (label_mask 쓰면 차원오류 발생)
                                        
                                        softamx를 취하기 전 Masking을 하는데, ae = ae.masked_fill(mask == 0, -1e10)
                                        
                                        이 때 불어(train) -> 영어(label) 번역이라 label_len, train_len 가 다르며, 
                                        
                                        보통 불어가 더 김 -> train_mask_size > label_mask_size
                                                                                  
                                        [1,1,27,30] : 1개의 sample, 1개의 head 에 대한 attention energey size = [27,30]

                                        [1,1,1,30] : 1개의 sample, 1개의 head, 1개의 단어에 대한 mask

                                        Query * key 곱하면, key가 항상 뒤라서 결과값이 key 차원의 영양을 받음

                                        Query * key = [128,8,27,30] : 128개의 sample, 8개의 head , 27 label에 대한 30 train attention score

                                        decoder의 값(Query-label)이 encoder의 단어 정보들(keys)을 참조하는 것임으로 encoder에 대한 masking만 하면 됨

                                        그래서 label_mask 쓰면 오류가 나고, train_mask를 쓰면 오류가 나지 않는 것


   5) softmax(AE dim = -1 )           : AE 마지막 차원 = label_seq_len , 각 Query에 대한 key 값들의 attention probability vbvalue 의미

                                        attention score = [128,8,27,30]


   6) word info                       : 각 단어에 대한 확률값 * 각 단어의 실제값 = 각 단어가 갖는 word information 

                                        [128, 8, 27, 30] * [128, 8, 30, 32] = [128, 8, 27, 32] 



   7) Concatenation                   : 1개의 encoder/decoder layer는 h개의 head를 갖고, 위 연산은 각 head 들에 대한 병렬 연산 임으로

                                        multi-head의 head를 concatenation 함으로써 입력 차원의 크기와 동일하게 조정

                                        [128,8,27,32] -> [128,27,8,32] -> [128,27,256] 


   8) output layers                   : [ 128,27,256 ] ->  [128,27,256]

  



 < Decoder >

 * encoder input 

   - word embedding vector : raw_label_data = [128,27] -> embedding vector = [128, 27, 256]

     pos embedding vector  : raw_pos_data = [128, 27] -> embedding vector = [ 128, 27, 256 ]

     enc_output            : [ 128, 30, 256 ]

     dec_scale_factor      : word embedding vector에 곱하는것으로 보아 각 embedding vector에 차별성을 부여하기 위해 사용 
     
 < Decoder_layer >

 * Decoder_layer input 

   - dencoder's 결과물 word_embedding_vector * encoder_scale_factor ) + pos_embedding_vector = [128,27,256]

 * flow 

   - init -> Masked Decdoer self Attention -> Residual & Norm -> Encoder Decoder Attention -> Residual & Norm -> simple ff layers -> Residual & Norm

     [128, 27, 256] -> [128,27,256] -> [128,27,256] -> [128,27,256] -> [128,27,256] -> [128,27,256]


 < Transformer >

 * train mask 

   - padding 으로 채워진 데이터만 걸러내기 


 * label mask 

   - label_sub_mask = torch.tril(torch.ones((label_len, label_len), device = self.device)).bool()

     torch.ones( [label_len, label_len] ) : 레이블 단어 길이 크기의 모든 원소의 값이 1인 2-d matrix

     torch.tril( a ).bool()               : matrix a의 하위 부분만 1로, 나머진 0으로 채워 하부 삼각형 형성 이 후, bool()에 의해 1인 위치만 True

        K
      Q 1 0 0 0 0 ...        row    : Query
        1 1 0 0 0 ...        colun  : Key 
        1 1 1 0 0 ...
        1 1 1 1 0 ...
        1 1 1 1 1 ...


     => 해당 mask는 Masked Decoder Self Attention 에서만 사용 ( encoder - decoder attention 에서는 train_mask )

        최종 label_mask 가 sub_mask & pad_mask 로 구성되는데, 1이 아니면 모두 False 이며 , False 일 경우 무시

        ∴ Query, key가 모두 label value 임으로 이러한 마스크 갖는 의의는 현 시점(t) 보다 미래 시점(t+k)에 있는 label 값과의 연관성은 고려하지 않겠다는 의미 

     

 * 구현 하면서 생각했던 점

   1) Multi-head attention 에서 linear layer를 통한 query, key, value vector 만들 때, 바로 head_dim = embed_dim / n_head 로 설정하면 안되나 ?

      => 1개의 encdoer/decoder에는 n개의 encoder/decoder layer가 있고 

         1개의 encoder_layer / decoder_layer에는 h개의 head가 있지만

         head 연산 과정에서는 weight가 필요한 NN이 관여 하는 것이 아니라, 

         단순히 행렬 곱을 통한 query에 대한 key의 attention score를 계산하는 것임으로 

         embed_dim -> head_dim 으로 direct 하게 설정하면, 여러개의 Query, key , value를 형성 해야 함으로

         embed_dim -> embed_dim 방식을 취해, 하나의 행렬 내에서 permute 연산을 통해 계산 하는것이 좀 더 쉽게 구현이 가능

         단, permute()로 데이터의 shape을 변경할 때는 contigious() 설정이 필요함 , transpose 와 같은 연산을 진행할떄 데이터가 올바른 위치에 저장 되도록 함


    2) reshape, permute

       * reshape : 데이터의 형태(차원)를 변화 시키고, row -> column 순으로 차례대로 데이터 집어 넣기 

       * permute : 데이터의 형태 및 해당 index에 존재하는 데이터를 동시에 변환 

                   행렬 곱을 위한 matrix 의 형 변환에 적합한 것은 permute 이다 
        

'''


import torch
import torch.nn as nn

class Multi_head_Attention(nn.Module):
    
    def __init__(self, embed_dim, n_head, dropout_ratio, device):

        super().__init__()

        assert embed_dim % n_head == 0 

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head

        # make Query, Key, Value vector
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # multi-head output dim = input_seq_data_dim
        self.output = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_ratio)

        self.scale_factor = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, src2query, src2key, src2value, mask = None):

        # init data dim
        # src2query = [128,30,256] , src2key = [128,30,256] , src2value = [128,30,256] in Encoder self attention
        # src2query = [128,27,256] , src2key = [128,27,256] , src2value = [128,27,256] in Masked decoder self attention
        # src2query = [128,27,256] , src2key = [128,30,256] , src2value = [128,30,256] in Encoder decoder attention
        
        batch_size = src2query.shape[0]

        # make query, key, value vectors
        # query = [128,30,256] , key = [128,30,256] , value = [128,30,256] in Encoder self attention
        # query = [128,27,256] , key = [128,27,256] , value = [128,27,256] in Masked decoder self attention
        # query = [128,27,256] , key = [128,30,256] , value = [128,30,256] in Encoder decoder attention
        query = self.q_linear(src2query)
        key = self.k_linear(src2key)
        value = self.k_linear(src2value)
        
        # n개의 head로 (qeury, key, value) vector 쌍 분할
        # query = [128,8,30,32] , key = [128,8,30,32] , value = [128,8,30,32] in Encoder self attention
        # query = [128.8,27,32] , key = [128.8,27,32] , value = [128.8,27,32] in Masked decoder self attention
        # query = [128,8,27,32] , key = [128,8,30,32] , value = [128,8,30,32] in Encoder decoder attention
        query = query.view(batch_size, -1, self.n_head, self.head_dim).permute(0,2,1,3)
        key   = key.view(batch_size, -1, self.n_head, self.head_dim).permute(0,2,1,3)
        value = value.view(batch_size, -1, self.n_head, self.head_dim).permute(0,2,1,3)
        
        # Attention Energy 
        # ae = [128,8,30,30] in Encoder self attention
        # ae = [128,8,27,27] in Masked decoder self attention
        # ae = [128,8,27,30] in Encoder decoder attention
        ae = torch.matmul( query, key.permute(0,1,3,2) ) / self.scale_factor
        
        # mask apply
        if mask is not None :  ae = ae.masked_fill(mask == 0, -1e10)

        # attention_score
        # attn_score = [128,8,30,30]  in Encoder self attention
        # attn_score = [128,8,27,27] in Masked decoder self attention
        # attn_score = [128,8,27,30] in Encoder decoder attention
        atten_score = torch.softmax(ae, dim = -1)
        
        # 각 단어별 연관성의 정도      
        # word_info = [128,8,30,32] in Encoder self attention
        # word_info = [128,8,27,32] in Masked decoder self attention
        # word_info = [128,8,27,32] in Encoder decoder attention
        word_info = torch.matmul(self.dropout(atten_score), value)
        
        # Concatenation  
        # contiguous : Contiguous = True 인 상태로 메모리 상 저장 구조 변경, 주로 transpose, permute와 같은 연산과 함께 사용 
        # word_info = [128,8,30,32] -> [128,30,8,32] -> [128,30,256] in Encoder self attention
        # word_info = [128,8,27,32] -> [128,27,8,32] -> [128,27,256] in Masked decoder self attention
        # word_info = [128,8,27,32] -> [128,27,8,32] -> [128,27,256] in Encoder decoder attention
        word_info = word_info.permute(0,2,1,3).contiguous()
        word_info = word_info.view(batch_size, -1, self.embed_dim)

        # encoder_output = [128, 30, 256] in Encoder self attention
        # encoder_output = [128, 27, 256] in Masked decoder self attention
        # encoder_output = [128, 27, 256] in Encoder decoder attention
        encoder_output = self.output(word_info)

        return encoder_output, atten_score


class simple_ff(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, dropout_ratio):

        super().__init__()

        self.pe_in = nn.Linear(embed_dim, ff_embed_dim)
        self.pe_out = nn.Linear(ff_embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        

    def forward(self, x):

        x = self.dropout(torch.relu(self.pe_in(x)))

        x = self.pe_out(x)

        return x 


class Encoder_layer(nn.Module):

    def __init__(self, embed_dim, n_head, ff_embed_dim, dropout_ratio, device):
        
        super().__init__()

        self.dropout = nn.Dropout(dropout_ratio)

        # encoder_layer 동작 순서에 맡게 선언
        self.encoder_self_attention = Multi_head_Attention(embed_dim, n_head, dropout_ratio, device)
        self.atten_norm = nn.LayerNorm(embed_dim)

        self.enc_ff = simple_ff(embed_dim, ff_embed_dim, dropout_ratio)
        self.enc_ff_norm = nn.LayerNorm(embed_dim)


    def forward(self, data, data_mask):
        
        # init : data = [128,30,256], data_mask = [ 128,1, 1, 30 ] 
        # _data = [128,30,256]
        _data, _ = self.encoder_self_attention( data, data, data, data_mask)
        
        # data = [128,30,256]
        # self attention residual connection & normalization
        data = self.atten_norm(data + self.dropout(_data) )
        
        # ff layer , _data = [128,30,256]
        _data = self.enc_ff(data)
        
        # ff residual& normalization, data = [128,30,256]
        data = self.enc_ff_norm(data + self.dropout(_data))
        

        return data
  

class Encoder(nn.Module):

    def __init__(self, input_dim, embed_dim, n_layers, n_head, ff_embed_dim, dropout_ratio, device, max_len = 100 ):
        
        super().__init__()
        
        self.enc_scale_factor = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
                                
        self.device = device
        self.dropout = nn.Dropout(dropout_ratio)

        # encoder 동작 순서대로 선언
        self.word_embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)

        # 1 encoder has n encoder_layers
        self.en_layers = nn.ModuleList( [Encoder_layer(embed_dim, n_head, ff_embed_dim, dropout_ratio, device) for _ in range(n_layers) ] )

        

    def forward(self, x, x_mask):
        
        # x = [128, 30]
        batch_size, seq_len = x.shape[0], x.shape[1]

        # pos_vec = [128,30]
        pos_vec = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size,1).to(self.device)
        
        # positional encoder + word embeding = [128,30,256]
        x = self.dropout( (self.word_embedding(x) * self.enc_scale_factor ) + self.pos_embedding(pos_vec) ) 
        
        # encoder layer 중첩사용, 마지막 encoder layers의 출력값 return 
        for en_layer in self.en_layers:

            x = en_layer(x , x_mask)


        return x


class Decoder_layer(nn.Module):

    def __init__(self, embed_dim, n_head, ff_embed_dim , dropout_ratio, device):
        
        super().__init__()

        self.dropout = nn.Dropout(dropout_ratio)
        self.device = device

        # decoder_layer 동작 순서에 맡게 선언
        self.masked_self_decoder_attention = Multi_head_Attention(embed_dim, n_head, dropout_ratio, device)
        self.mask_norm = nn.LayerNorm(embed_dim)

        self.encoder_decoder_attention = Multi_head_Attention(embed_dim, n_head, dropout_ratio, device)
        self.enc_dec_norm = nn.LayerNorm(embed_dim)

        self.dec_ff = simple_ff(embed_dim, ff_embed_dim, dropout_ratio)
        self.dec_ff_norm = nn.LayerNorm(embed_dim)

    def forward(self, label, enc_output, label_mask, enc_mask):

        # decoder init data shape
        # enc_output = [128,30,256] , label = [128,27], enc_mask = [128,1,1,30], label_mask = [128,1,27,27] 
        
        # Maksed self decoder attention
        # _label = [128, 27, 256] 
        _label, _ = self.masked_self_decoder_attention(label, label, label, label_mask)
        
        # Residual & Normalization
        # label = [128, 27, 256] 
        label = self.mask_norm(label + self.dropout(_label))
        
        # encdoer_decoder attention
        # label 과 enc_output 의 sequence가 다를 수 있음  => < 해결법 >
        
        # _label = [128,27,256]
        _label, atten_score = self.encoder_decoder_attention(label, enc_output, enc_output, enc_mask)
        
        # Residual & Normalization, label = [128,27,256]
        label = self.enc_dec_norm(label + self.dropout(_label))
        
        # ff layer , _label = [128,27,256]
        _label = self.dec_ff(label)
        
        # ff residual& normalization, label = [128,27,256]
        label = self.dec_ff_norm(label + self.dropout(_label))
        
        return label, atten_score
  

class Decoder(nn.Module):

    def __init__(self, decoder_input_dim, embed_dim, n_layers, n_head, ff_embed_dim, dropout_ratio, device, max_len=100):
        
        super().__init__()

        self.dec_scale_factor = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
        self.device = device
        self.dropout = nn.Dropout(dropout_ratio)


        # decoder 동작 순서대로 선언
        self.decoder_word_embedding = nn.Embedding(decoder_input_dim, embed_dim)
        
        # 위치는 상대적임으로, 최대 길이를 기준 pos vector 생성 
        self.decoder_pos_embedding = nn.Embedding(max_len, embed_dim)

        self.dec_layers = nn.ModuleList([ Decoder_layer(embed_dim, n_head, ff_embed_dim , dropout_ratio, device) for _ in range(n_layers) ])

        self.dec_output = nn.Linear(embed_dim, decoder_input_dim)



    def forward(self, label, enc_output, label_mask, enc_mask):

        # label = [128,27] , enc_output = [128, 30, 256]
        batch_size, seq_len = label.shape[0], label.shape[1]

        # pos_vec = [128, 27]
        pos_vec = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size,1).to(self.device)
        
        # positional encoder + word embeding
        # label_dim = [128, 27, 256] 
        label = self.dropout( (self.decoder_word_embedding(label) * self.dec_scale_factor ) + self.decoder_pos_embedding(pos_vec) ) 
        
        # decoder layer 중첩사용, 마지막 decoder layers의 출력값 return 
        # label = [128, 27, 256] , attention = [128,8,27,30] 
        for dec_layer in self.dec_layers:

            label, attention = dec_layer(label, enc_output, label_mask, enc_mask)
        
        # decoder_output = [128,27,5893]
        decoder_output = self.dec_output(label)
        
        return decoder_output, attention


class Transformer(nn.Module):

    def __init__(self, encoder, decoder, train_pad_idx, label_pad_idx, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.train_pad_idx = train_pad_idx
        self.label_pad_idx = label_pad_idx
        self.device = device

    def make_train_mask(self, train):

        # train = [128,30]
        
        # train_mask = [128,1,1,30] = [[[ True, True, ... False]]] 형태
        train_mask = (train != self.train_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return train_mask

    def make_label_mask(self, label):

        # label = [128, 26] => label[:, :-1] 
        # label에서 마지막 <eos> token 제외하고 입력으로 들어와서 label_len = 27-1 = 26
        label_len = label.shape[1] 
        
        # label_pad_mask = [128,1,1,26]
        label_pad_mask = (label != self.label_pad_idx).unsqueeze(1).unsqueeze(2)

        
        # torch.tril : 하부 삼각형(위쪽은 0 ) tensor 생성 -> ones임으로 모두 1 -> bool() 했으니 all True
        # Transformer -> label mask 참조 
        label_sub_mask = torch.tril(torch.ones((label_len, label_len), device = self.device)).bool()
             
        # label_mask = [128,1, 26,26]
        label_mask = label_pad_mask & label_sub_mask

        return label_mask


    def forward(self, train, label):
        
        train_mask = self.make_train_mask(train)
        label_mask = self.make_label_mask(label)
        
        enc_output = self.encoder(train, train_mask)

        # 일단 여기 굳이 train_mask 필요 없음 
        # decoder_output dim = [128,27,5893] , attention = [128,8,27,30] 
        decoder_output, attention = self.decoder(label, enc_output, label_mask, train_mask)
        
        return decoder_output, attention






