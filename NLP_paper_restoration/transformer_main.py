# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-13


< Transformer 복원하기  >

 - Attention is All You Need (NIPS 2017)


 * 특징

  1. 여러개의 Encoder, decoder를 사용 ( 논문과 실제 설계에서의 차이점 )

     => 실제 설계에서 encoder, decoder는 각각의 src, trg sequence에 대한 word embedding 및 Positional Encoding 만 진행하고,  ( RNN 계열 모델을 사용하지 않아 seq 에 대한 order information 필요 )

        Multi-head attention, Residual Learning 과 같은 기능은 encoder/decoder layers class를 통해 진행

        * 각 encoder / decoder layer 들은 layer 끼리 서로 다른 params를 갖는다

        * Residual Learning : 특정 layer를 건너 뛰어, 복사가 된 값을 그대로 넣어주는 것을 의미 ( 복사 된 값임으로, 학습 난이도가 낮고, 수렴 속도가 빠르기 떄문에 global optimal에 도달하기 쉬움 )

        * Encoder has 1 type Multi-head Attention  ( Encoder self attention )

        * Decoder has 2 type Multi-head Attentions ( Masked decoder self attention , Encoder - Decoder attention )




  2. Multi-head Attention 

     => Query(물어보는 주체) , key(물어보는 대상), value를 이용하여 query 와 key 가 얼마나 연관성이 있는지 게산

        * i'm a teacher 에 포함되는 각 단어 i , am , a , teacher 가 서로 얼마나 연관성이 있는지 계산 ( in encoder self attention )

          ex) Query(I), keys = [ am, a , teacher ] 



     => 동작 방식 
     
        1. input_data = src 가 3개로 복제  =>  query, key, value

        2. Linear layer를 통과 후, h 개의 (query, key, value) 쌍을 만듬   
        
           =>  입력 문장에 대해서 h개의 서로 다른 attention concept 을 학습하여 다양한 특징들을 학습할 수 있게 함

               첫번째 단계에서 복제되는 src는 h개가 모두 동일하겠지만, 이후 linear layer를 거쳐 (query, key, value) 쌍을 만들기 떄문에 각 head 마다 다른 값의 쿼리가 형성 

               즉, 1 encoder/decoder has N encoder/decoder layers 

                   1 encoder_layer has h attentions  , 1 decoder_layers has 2*h attentions (decoder has 2 types M-H attentions)

  
        3. Sclaed Dot-product Attention

           =>  Q, K , V : 각각 Linear layer를 통해 만들어지는 query, key , value vector( 8-dimension )  라면

               1. Attention Energy : Q * K^T   query 와 key 에 대한 내적값 계산 

               2. Normalization :   (Q * K^T ) / sqrt(d_key)  ,  sqrt(d_key) : key vector dim = 256 의 루트 값 

                                    => softmax의 입력으로 normalization 없이 사용되면 값이 너무 클 경우 softmax의 양끝값으로 배치되며 이는 Vanishing Gradient 발생

                                      d_key := key_vector의 차원 , 즉 벡터의 길이만큼 나눠줌으로써 0에 근사시키고자 함

                                       < so, why root ? >

                                       => 두 vector의 내적에서  각 요소들의 곱은 음수, 양수 모두 들쭉 날쭉하기 떄문에 내적값이 꼭 길이에 비례하여 커지는것이 아니기 떄문

                                          실제로 sqrt(x) 그래프와 sqrt(randn(n,1)^T@randn(n,1)) 그래프가 비스무리하게 나오는것을 알 수 있음  
                                           
                                          실험으로 그려보면 루트 보다는 0.46 승으로 하면 그래프가 일치한것을 볼 수 있음 




               3. Attention_score :  softmax((Q * K^T ) / sqrt(d_key))

    

        4. 각 head에서 계산된 attention value Concatenation

           => word embedding 된 input_seq_dim = 64 , n_head = 8  이라면, 각 head의 scaled_dot_product attention 과정에 사용 되는 

              query, key, value vector dim = word_embeding_dim / n _head =  64 / 8 = 8 dimension 으로 각각 설정 된다.

              이 후, 각 head의 결과들을 concatenation 하면,  처음의 embedding_dim = 64와 같아진다

              why ?

              => decoder와 encoder의 word embedding dimension은 보통 동일하게 설정 되며,  decoder 의 encoder-decoder attention layer에 

                 입력으로 사용되어야 하기 때문에 차원을 맞춰주는 것으로 추측 된다.



     => Transformer Attention 종류 

        * Encoder self attention           : 입력 단어가 서로 query, key 가 됨 => 입력 단어가 서로 어떠한 영향을 갖고 있는지 계산 ( 문맥에 대한 정보를 학습 )

        * Masked decoder self attention    : 현 시점을 t라고 했을 때, t-k 시점(과거 시점)의 값들과 attention 수행 ( 미래를 예측하는 task에서 미래 단어와 attention 하는 것은 옳지 않음 )

        * Encoder - Decoder attention      : encoder's last layer의 output = keys, decoder를 통해 생성된 단어 : query  



  3. Positional Encoding (PE)
       
     * 정해진 주기 함수(sin, cos)를 활용한 공식을 사용하여 입력 문장의 order information 생성 ( in paper )

        * PE(pos, 2i) = sin( pos/10000^(2i/d_model) )

        * PE(pos, 2i+1) = cos( pos/10000^(2i/d_model) )


     * 입력 문장에서 각 단어의 상대적 위치를 encoder/decoder가 알 수 있게만 하면 된다.

       => 즉, 별도의 position embedding layer를 활용 (학습이 가능하기 때문)




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

# model 선언시 to(device) 주의 
model = Transformer(encoder, decoder, train_pad_idx, label_pad_idx, device).to(device)


# paper 에 따른 model 가중치 params 초기화 
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
        
        # 매 iteration 마다 optim 초기화 
        optimizer.zero_grad()
        
        # label[:,:-1]출력 단어의 마지막 <eos> 제외 
        # output = [128,27,5893]
        output, _ = model(x_train, label[:,:-1])
        
        # output은 3D tensor 이며, [-1] 은 마지막 차원을 의미함으로 , output_dim = [5893]
        output_dim = output.shape[-1]
        
        # output = [3456,5893] -> label과의 loss를 한번에 계산하기 위해 batch * 단어 수 하나로 묶어버림
        output = output.contiguous().view(-1, output_dim)
        
        # 출력 단어의 <sos> token 제외 
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
    
    # 평가모드
    model.eval()

    val_loss = 0 

    with torch.no_grad():

        for idx, batch in enumerate(iterator):

            x_eval = batch.src
            label = batch.trg

            # 평가 시 teacher forcing 사용 x
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

