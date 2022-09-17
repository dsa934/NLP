# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-15


< Transformer Model's dimension chnage Analysis >

 - �Է� �����Ͱ� transformer model�� �����ϴ� �� class�� ��� �ϸ鼭 ��ȭ�ϴ� ������ ���������� ����

 - Transforemr -> Encoder -> Encoder layers -> Multi-head attention -> decoder -> decoder_layers -> Multi-head attention



 < Transformer >

 * transformer input 

   - train / label data
    
     train = [128,30]  = [batch, train_seq_len]  ,label = [128,27]  = [batch, label_seq_len] 
  
     => label�� <eos> token�� �����ϰ� transformer�� �Է� �����ͷ� Ȱ�� ������, ���� label_seq_len -1 
     
 * train, label mask

   -  train_mask = [128,1,1,30] = [batch_size, 1, 1, train_seq_len] => 128���� ����, 1���� head , 1���� query �ܾ ���� 30���� key ���� ���� mask

      label_mask = [128,1,1,26] = [batch_size, 1, 1, (label_seq_len -1 ) ]



 * masked train / label data ���


 < Encoder >

 * encoder input 

   - word embedding vector : raw_train_data = [128,30] -> embedding vector = [128, 30, 256]

     pos embedding vector  : raw_pos_data = [128,30] -> embedding vector = [ 128, 30, 256 ]

     enc_scale_factor      : word embedding vector�� �������� �ο��ϱ� ���� ��� 

     
 < Encoder_layer >

 * Encoder_layer input 

   - encoder's ����� word_embedding_vector * encoder_scale_factor ) + pos_embedding_vector = [128,30,256]


 * flow 

   - init -> Encdoer self Attention -> Residual add & Norm -> simple ff layers -> Residual add & Norm

     [128, 30, 256] -> [128,30,256] -> [128,30,256] -> [128,30,256] -> [128,30,256] 



 < Multi - head Attention �� >

 * Encoder self attention 
 
   1) encoder self atten input dim    : 3 train_embedding_vectors = [128,30,256] x 3 , train_mask = [128, 1, 1, 30]

   2) Q, k, V vector �����           : Query / Key / Value vectors = [128, 30, 256]

   3) n���� head�� vector �� ����     : query/key/value. veiw(batch, -1, n_head, head_dim) . permute(0,2,1,3)

                                        [128,30,256] -> [128, 30, 8, 32] -> [ 128, 8, 30, 32] = [ batch, n_head, train_seq_len, head_dim]
    
   4) Attention Energy(AE)            : torch.matmul(Query, Key) : 3���� �̻� tensor ��İ� [b,n a,b] * [b, n, b, c] = [b, n, a, c]
   
                                        torch.matmul(query, Key.permute(0,1,3,2))

                                        [128,8,30,32] * [ 128, 8, 32, 30 ] = [128, 8, 30, 30] 

   5) softmax(AE dim = -1 )           : AE ������ ���� = train_seq_len , �� Query�� ���� key ������ attention probability vbvalue �ǹ�

                                        attention score = [128,8,30,30]

   6) word info                       : �� �ܾ ���� Ȯ���� * �� �ܾ��� ������ = �� �ܾ ���� word information

                                        [128, 8, 30, 30] * [128, 8, 30, 32] = [128, 8, 30, 32]

   7) Concatenation                   : 1���� encoder/decoder layer�� h���� head�� ����, �� ������ �� head �鿡 ���� ���� ���� ������

                                        multi-head�� head�� concatenation �����ν� �Է� ������ ũ��� �����ϰ� ����

                                        [128,8,30,32] -> [128,30,8,32] -> [ 128,30, 256] 

   8) output layers                   : [ 128,30,256 ] ->  [128,30,256]



 * Masked Decoder Self Attention 

   [128,30,256] -> [128,30,256] -> [128.8,27,32] -> [128,8,27,27] -> [128,8,27,32] -> [128,27,256] -> [128,27,256]  
 

 * Encoder - Decoder  attention 

   1) Encoder decoder atten dim       : label = [128,27,256] , enc_output = [128,30,256], enc_output = [128,30,256]

   2) Q, k, V vector �����           : Query = [128,27,256] , Key = [128,30,256]  Value = [128,30,256]

   3) n���� head�� vector �� ����     : query/key/value. veiw(batch, -1, n_head, head_dim) . permute(0,2,1,3)

                                        Query : [ batch, n_head, label_seq_len, head_dim] = [128,8,27,32] 

                                        K & V : [ batch, n_head, train_seq_len, head_dim] = [128,8,30,32] 

   4) Attention Energy(AE)            : torch.matmul(query, Key.permute(0,1,3,2))

                                        query = [128, 8, 27, 32], key = [128, 8, 32, 30]

                                        [128,8,27,32] * [ 128, 8, 32, 30 ] = [128, 8, 27, 30] 

                                      * Encoder-decoder Attention���� encoder_mask�� ��� �Ǵ� ���� (label_mask ���� �������� �߻�)
                                        
                                        softamx�� ���ϱ� �� Masking�� �ϴµ�, ae = ae.masked_fill(mask == 0, -1e10)
                                        
                                        �� �� �Ҿ�(train) -> ����(label) �����̶� label_len, train_len �� �ٸ���, 
                                        
                                        ���� �Ҿ �� �� -> train_mask_size > label_mask_size
                                                                                  
                                        [1,1,27,30] : 1���� sample, 1���� head �� ���� attention energey size = [27,30]

                                        [1,1,1,30] : 1���� sample, 1���� head, 1���� �ܾ ���� mask

                                        Query * key ���ϸ�, key�� �׻� �ڶ� ������� key ������ ������ ����

                                        Query * key = [128,8,27,30] : 128���� sample, 8���� head , 27 label�� ���� 30 train attention score

                                        decoder�� ��(Query-label)�� encoder�� �ܾ� ������(keys)�� �����ϴ� �������� encoder�� ���� masking�� �ϸ� ��

                                        �׷��� label_mask ���� ������ ����, train_mask�� ���� ������ ���� �ʴ� ��


   5) softmax(AE dim = -1 )           : AE ������ ���� = label_seq_len , �� Query�� ���� key ������ attention probability vbvalue �ǹ�

                                        attention score = [128,8,27,30]


   6) word info                       : �� �ܾ ���� Ȯ���� * �� �ܾ��� ������ = �� �ܾ ���� word information 

                                        [128, 8, 27, 30] * [128, 8, 30, 32] = [128, 8, 27, 32] 



   7) Concatenation                   : 1���� encoder/decoder layer�� h���� head�� ����, �� ������ �� head �鿡 ���� ���� ���� ������

                                        multi-head�� head�� concatenation �����ν� �Է� ������ ũ��� �����ϰ� ����

                                        [128,8,27,32] -> [128,27,8,32] -> [128,27,256] 


   8) output layers                   : [ 128,27,256 ] ->  [128,27,256]

  



 < Decoder >

 * encoder input 

   - word embedding vector : raw_label_data = [128,27] -> embedding vector = [128, 27, 256]

     pos embedding vector  : raw_pos_data = [128, 27] -> embedding vector = [ 128, 27, 256 ]

     enc_output            : [ 128, 30, 256 ]

     dec_scale_factor      : word embedding vector�� ���ϴ°����� ���� �� embedding vector�� �������� �ο��ϱ� ���� ��� 
     
 < Decoder_layer >

 * Decoder_layer input 

   - dencoder's ����� word_embedding_vector * encoder_scale_factor ) + pos_embedding_vector = [128,27,256]

 * flow 

   - init -> Masked Decdoer self Attention -> Residual & Norm -> Encoder Decoder Attention -> Residual & Norm -> simple ff layers -> Residual & Norm

     [128, 27, 256] -> [128,27,256] -> [128,27,256] -> [128,27,256] -> [128,27,256] -> [128,27,256]


 < Transformer >

 * train mask 

   - padding ���� ä���� �����͸� �ɷ����� 


 * label mask 

   - label_sub_mask = torch.tril(torch.ones((label_len, label_len), device = self.device)).bool()

     torch.ones( [label_len, label_len] ) : ���̺� �ܾ� ���� ũ���� ��� ������ ���� 1�� 2-d matrix

     torch.tril( a ).bool()               : matrix a�� ���� �κи� 1��, ������ 0���� ä�� �Ϻ� �ﰢ�� ���� �� ��, bool()�� ���� 1�� ��ġ�� True

        K
      Q 1 0 0 0 0 ...        row    : Query
        1 1 0 0 0 ...        colun  : Key 
        1 1 1 0 0 ...
        1 1 1 1 0 ...
        1 1 1 1 1 ...


     => �ش� mask�� Masked Decoder Self Attention ������ ��� ( encoder - decoder attention ������ train_mask )

        ���� label_mask �� sub_mask & pad_mask �� �����Ǵµ�, 1�� �ƴϸ� ��� False �̸� , False �� ��� ����

        �� Query, key�� ��� label value ������ �̷��� ����ũ ���� ���Ǵ� �� ����(t) ���� �̷� ����(t+k)�� �ִ� label ������ �������� ������� �ʰڴٴ� �ǹ� 

     

 * ���� �ϸ鼭 �����ߴ� ��

   1) Multi-head attention ���� linear layer�� ���� query, key, value vector ���� ��, �ٷ� head_dim = embed_dim / n_head �� �����ϸ� �ȵǳ� ?

      => 1���� encdoer/decoder���� n���� encoder/decoder layer�� �ְ� 

         1���� encoder_layer / decoder_layer���� h���� head�� ������

         head ���� ���������� weight�� �ʿ��� NN�� ���� �ϴ� ���� �ƴ϶�, 

         �ܼ��� ��� ���� ���� query�� ���� key�� attention score�� ����ϴ� �������� 

         embed_dim -> head_dim ���� direct �ϰ� �����ϸ�, �������� Query, key , value�� ���� �ؾ� ������

         embed_dim -> embed_dim ����� ����, �ϳ��� ��� ������ permute ������ ���� ��� �ϴ°��� �� �� ���� ������ ����

         ��, permute()�� �������� shape�� ������ ���� contigious() ������ �ʿ��� , transpose �� ���� ������ �����ҋ� �����Ͱ� �ùٸ� ��ġ�� ���� �ǵ��� ��


    2) reshape, permute

       * reshape : �������� ����(����)�� ��ȭ ��Ű��, row -> column ������ ���ʴ�� ������ ���� �ֱ� 

       * permute : �������� ���� �� �ش� index�� �����ϴ� �����͸� ���ÿ� ��ȯ 

                   ��� ���� ���� matrix �� �� ��ȯ�� ������ ���� permute �̴� 
        

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
        
        # n���� head�� (qeury, key, value) vector �� ����
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
        
        # �� �ܾ �������� ����      
        # word_info = [128,8,30,32] in Encoder self attention
        # word_info = [128,8,27,32] in Masked decoder self attention
        # word_info = [128,8,27,32] in Encoder decoder attention
        word_info = torch.matmul(self.dropout(atten_score), value)
        
        # Concatenation  
        # contiguous : Contiguous = True �� ���·� �޸� �� ���� ���� ����, �ַ� transpose, permute�� ���� ����� �Բ� ��� 
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

        # encoder_layer ���� ������ �ð� ����
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

        # encoder ���� ������� ����
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
        
        # encoder layer ��ø���, ������ encoder layers�� ��°� return 
        for en_layer in self.en_layers:

            x = en_layer(x , x_mask)


        return x


class Decoder_layer(nn.Module):

    def __init__(self, embed_dim, n_head, ff_embed_dim , dropout_ratio, device):
        
        super().__init__()

        self.dropout = nn.Dropout(dropout_ratio)
        self.device = device

        # decoder_layer ���� ������ �ð� ����
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
        # label �� enc_output �� sequence�� �ٸ� �� ����  => < �ذ�� >
        
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


        # decoder ���� ������� ����
        self.decoder_word_embedding = nn.Embedding(decoder_input_dim, embed_dim)
        
        # ��ġ�� �����������, �ִ� ���̸� ���� pos vector ���� 
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
        
        # decoder layer ��ø���, ������ decoder layers�� ��°� return 
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
        
        # train_mask = [128,1,1,30] = [[[ True, True, ... False]]] ����
        train_mask = (train != self.train_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return train_mask

    def make_label_mask(self, label):

        # label = [128, 26] => label[:, :-1] 
        # label���� ������ <eos> token �����ϰ� �Է����� ���ͼ� label_len = 27-1 = 26
        label_len = label.shape[1] 
        
        # label_pad_mask = [128,1,1,26]
        label_pad_mask = (label != self.label_pad_idx).unsqueeze(1).unsqueeze(2)

        
        # torch.tril : �Ϻ� �ﰢ��(������ 0 ) tensor ���� -> ones������ ��� 1 -> bool() ������ all True
        # Transformer -> label mask ���� 
        label_sub_mask = torch.tril(torch.ones((label_len, label_len), device = self.device)).bool()
             
        # label_mask = [128,1, 26,26]
        label_mask = label_pad_mask & label_sub_mask

        return label_mask


    def forward(self, train, label):
        
        train_mask = self.make_train_mask(train)
        label_mask = self.make_label_mask(label)
        
        enc_output = self.encoder(train, train_mask)

        # �ϴ� ���� ���� train_mask �ʿ� ���� 
        # decoder_output dim = [128,27,5893] , attention = [128,8,27,30] 
        decoder_output, attention = self.decoder(label, enc_output, label_mask, train_mask)
        
        return decoder_output, attention






