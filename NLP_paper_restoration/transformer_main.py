# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-13


< Transformer �����ϱ�  >

 - Attention is All You Need (NIPS 2017)


 * Ư¡

  1. �������� Encoder, decoder�� ��� ( ���� ���� ���迡���� ������ )

     => ���� ���迡�� encoder, decoder�� ������ src, trg sequence�� ���� word embedding �� Positional Encoding �� �����ϰ�,  ( RNN �迭 ���� ������� �ʾ� seq �� ���� order information �ʿ� )

        Multi-head attention, Residual Learning �� ���� ����� encoder/decoder layers class�� ���� ����

        * �� encoder / decoder layer ���� layer ���� ���� �ٸ� params�� ���´�

        * Residual Learning : Ư�� layer�� �ǳ� �پ�, ���簡 �� ���� �״�� �־��ִ� ���� �ǹ� ( ���� �� ��������, �н� ���̵��� ����, ���� �ӵ��� ������ ������ global optimal�� �����ϱ� ���� )

        * Encoder has 1 type Multi-head Attention  ( Encoder self attention )

        * Decoder has 2 type Multi-head Attentions ( Masked decoder self attention , Encoder - Decoder attention )




  2. Multi-head Attention 

     => Query(����� ��ü) , key(����� ���), value�� �̿��Ͽ� query �� key �� �󸶳� �������� �ִ��� �Ի�

        * i'm a teacher �� ���ԵǴ� �� �ܾ� i , am , a , teacher �� ���� �󸶳� �������� �ִ��� ��� ( in encoder self attention )

          ex) Query(I), keys = [ am, a , teacher ] 



     => ���� ��� 
     
        1. input_data = src �� 3���� ����  =>  query, key, value

        2. Linear layer�� ��� ��, h ���� (query, key, value) ���� ����   
        
           =>  �Է� ���忡 ���ؼ� h���� ���� �ٸ� attention concept �� �н��Ͽ� �پ��� Ư¡���� �н��� �� �ְ� ��

               ù��° �ܰ迡�� �����Ǵ� src�� h���� ��� �����ϰ�����, ���� linear layer�� ���� (query, key, value) ���� ����� ������ �� head ���� �ٸ� ���� ������ ���� 

               ��, 1 encoder/decoder has N encoder/decoder layers 

                   1 encoder_layer has h attentions  , 1 decoder_layers has 2*h attentions (decoder has 2 types M-H attentions)

  
        3. Sclaed Dot-product Attention

           =>  Q, K , V : ���� Linear layer�� ���� ��������� query, key , value vector( 8-dimension )  ���

               1. Attention Energy : Q * K^T   query �� key �� ���� ������ ��� 

               2. Normalization :   (Q * K^T ) / sqrt(d_key)  ,  sqrt(d_key) : key vector dim = 256 �� ��Ʈ �� 

                                    => softmax�� �Է����� normalization ���� ���Ǹ� ���� �ʹ� Ŭ ��� softmax�� �糡������ ��ġ�Ǹ� �̴� Vanishing Gradient �߻�

                                      d_key := key_vector�� ���� , �� ������ ���̸�ŭ ���������ν� 0�� �ٻ��Ű���� ��

                                       < so, why root ? >

                                       => �� vector�� ��������  �� ��ҵ��� ���� ����, ��� ��� ���� �����ϱ� ������ �������� �� ���̿� ����Ͽ� Ŀ���°��� �ƴϱ� ����

                                          ������ sqrt(x) �׷����� sqrt(randn(n,1)^T@randn(n,1)) �׷����� �񽺹����ϰ� �����°��� �� �� ����  
                                           
                                          �������� �׷����� ��Ʈ ���ٴ� 0.46 ������ �ϸ� �׷����� ��ġ�Ѱ��� �� �� ���� 




               3. Attention_score :  softmax((Q * K^T ) / sqrt(d_key))

    

        4. �� head���� ���� attention value Concatenation

           => word embedding �� input_seq_dim = 64 , n_head = 8  �̶��, �� head�� scaled_dot_product attention ������ ��� �Ǵ� 

              query, key, value vector dim = word_embeding_dim / n _head =  64 / 8 = 8 dimension ���� ���� ���� �ȴ�.

              �� ��, �� head�� ������� concatenation �ϸ�,  ó���� embedding_dim = 64�� ��������

              why ?

              => decoder�� encoder�� word embedding dimension�� ���� �����ϰ� ���� �Ǹ�,  decoder �� encoder-decoder attention layer�� 

                 �Է����� ���Ǿ�� �ϱ� ������ ������ �����ִ� ������ ���� �ȴ�.



     => Transformer Attention ���� 

        * Encoder self attention           : �Է� �ܾ ���� query, key �� �� => �Է� �ܾ ���� ��� ������ ���� �ִ��� ��� ( ���ƿ� ���� ������ �н� )

        * Masked decoder self attention    : �� ������ t��� ���� ��, t-k ����(���� ����)�� ����� attention ���� ( �̷��� �����ϴ� task���� �̷� �ܾ�� attention �ϴ� ���� ���� ���� )

        * Encoder - Decoder attention      : encoder's last layer�� output = keys, decoder�� ���� ������ �ܾ� : query  



  3. Positional Encoding (PE)
       
     * ������ �ֱ� �Լ�(sin, cos)�� Ȱ���� ������ ����Ͽ� �Է� ������ order information ���� ( in paper )

        * PE(pos, 2i) = sin( pos/10000^(2i/d_model) )

        * PE(pos, 2i+1) = cos( pos/10000^(2i/d_model) )


     * �Է� ���忡�� �� �ܾ��� ����� ��ġ�� encoder/decoder�� �� �� �ְԸ� �ϸ� �ȴ�.

       => ��, ������ position embedding layer�� Ȱ�� (�н��� �����ϱ� ����)




'''

from transformer_model import Multi_head_Attention, simple_ff, Encoder , Encoder_layer, Decoder, Decoder_layer, Transformer
import transformer_preprocessing 
import torch
import torch.nn as nn 
import torch.optim as optim

# set hyper params
src, trg, device = transformer_preprocessing.src , transformer_preprocessing.trg, transformer_preprocessing.device
train_iterator , val_iterator , test_iterator = transformer_preprocessing.train_iter , transformer_preprocessing.val_iter, transformer_preprocessing.test_iter

input_dim, output_dim = len(src.vocab), len(trg.vocab)
encoder_embed_dim, decoder_embed_dim = 256, 256
layers, heads, ff_dim, dropout_ratio = 3, 8, 512, 0.1

train_pad_idx = src.vocab.stoi[src.pad_token]
label_pad_idx = trg.vocab.stoi[trg.pad_token]

learning_rate = 0.0005
                
# model setup
encoder = Encoder(input_dim, encoder_embed_dim, layers, heads, ff_dim, dropout_ratio, device)
decoder = Decoder(output_dim , decoder_embed_dim, layers, heads, ff_dim, dropout_ratio, device)

# model ����� to(device) ���� 
model = Transformer(encoder, decoder, train_pad_idx, label_pad_idx, device).to(device)


# paper �� ���� model ����ġ params �ʱ�ȭ 
def count_parameters(model):
    return sum(_w.numel() for _w in model.parameters() if _w.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)



# optimizer set
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# ignore pad token 
criterion = nn.CrossEntropyLoss(ignore_index = label_pad_idx)



# traing function
def train(model, iterator, optimizer, criterion, clip):

    # train mode
    model.train()

    train_loss = 0
    
    for idx, batch in enumerate(iterator):

        # x_train = [128, 30] = [batch_size, train_seq_len] , label = [128,27] = [ batch_size, label_seq_len ]
        x_train = batch.src
        label = batch.trg
        
        # �� iteration ���� optim �ʱ�ȭ 
        optimizer.zero_grad()
        
        # label[:,:-1]��� �ܾ��� ������ <eos> ���� 
        # output = [128,27,5893]
        output, _ = model(x_train, label[:,:-1])
        
        # output�� 3D tensor �̸�, [-1] �� ������ ������ �ǹ������� , output_dim = [5893]
        output_dim = output.shape[-1]
        
        # output = [3456,5893] -> label���� loss�� �ѹ��� ����ϱ� ���� batch * �ܾ� �� �ϳ��� �������
        output = output.contiguous().view(-1, output_dim)
        
        # ��� �ܾ��� <sos> token ���� 
        # label = [3456]
        label = label[:,1:].contiguous().view(-1)
        
        loss = criterion(output, label)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        train_loss += loss.item()


    return train_loss / len(iterator)


# def evaluate
def evaluate(model, iterator, criterion):
    
    # �򰡸��
    model.eval()

    val_loss = 0 

    with torch.no_grad():

        for idx, batch in enumerate(iterator):

            x_eval = batch.src
            label = batch.trg

            # �� �� teacher forcing ��� x
            output, _ = model(x_eval, label[:,:-1])

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)

            label = label[:,1:].contiguous().view(-1)

            loss = criterion(output, label)

            val_loss += loss.item()

    return val_loss / len(iterator)


# training

import math 

n_epoch, clip = 10, 1

best_val_loss = float('inf')


for epoch in range(n_epoch):

    train_loss = train(model, train_iterator, optimizer, criterion, clip)

    val_loss = evaluate(model, val_iterator, criterion)

    if val_loss < best_val_loss :

        best_val_loss = val_loss
        torch.save(model.state_dict(), "translation_transformer.pt")
    
    print("==============")
    print(f" < epoch : {epoch+1} > ")
    print(f' Train Loss : {train_loss:.3f} | PerPlexity : {math.exp(train_loss):.3f}')
    print(f' Val Loss : {val_loss:.3f} | Val Perplexity : {math.exp(val_loss):.3f}')


# testing
model.load_state_dict(torch.load('translation_transformer.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f"Test loss : {test_loss:.3f} | Test PPL : {math.exp(test_loss): .3f}")

