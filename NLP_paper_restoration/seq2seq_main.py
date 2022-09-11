# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-03


< 간소화 Seq2Seq model 복원하기 >

 - Sequence to Sequence Learning with Neural Networks (NIPS 2014) 

 - 물리적 이슈 (개인 컴퓨터의 성능 한계)로 인해 실제 Seq2Seq논문에 사용된 <> dataset 대신 용량이 상대적으로 적은  Multi30k 데이터셋을 활용하여

   seq2seq model 복원해보기 


<Step>

1. 데이터 전처리

   - seq2seq_preprocessing.py


2. 모델 설계

   - seq2seq_model.py


3. Seq2Seq 복원

   1. encoder, decoder, seq2seq model 객체 선언 

   2. Parameter 초기화 ( 논문에서 언급 )

   3. train ,evaluate function set 

   4. training, testing



  cf : 구현시 참고 

   * 토큰화 된 데이터에 대하여 패딩 데이터는 무시 하기 


   * train, label 데이터의 단어 수 

     => 불어에서 영어로 번역하는 task 임으로, 
    
        입력으로 사용된 train data의 단어 수 와  정답으로 활용되는 label의 단어 수가 다를 수 있다.


   * clip 

     => RNN 계열 신경망에서는 Gradient Vanishing or exploding이 자주 발생하는데 이를 방지하기 위해 사용

        일정 임계값(Threshold)를 초과할 경우 clip(자르다, 유지)


   * CrossEntropy

     =>  output_dim = [ 3712, 5893 ]  : (30-1) * 128  , 30개의 단어 중 <sos> 를 제외하고, 128개의 batch_size에 존재하는 단어를 모두 나열

         label_dim = [ 3712 ]

         => 3712 개의 단어에 대하여 , 5893개의 단어 후보 들 중 가장 확률이 높은 단어를 찾기 위해 CrossEntropy 적용 




'''

import seq2seq_preprocessing 
from seq2seq_model import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn 
import torch.optim as optim

# init hyperparams
# src, trg 는 multi30k 데이터에서 이미 field column 값으로 설정 해서 변경이 불가능
src = seq2seq_preprocessing.src
trg = seq2seq_preprocessing.trg
device = seq2seq_preprocessing.device
train_iterator , val_iterator , test_iterator = seq2seq_preprocessing.train_iter , seq2seq_preprocessing.val_iter, seq2seq_preprocessing.test_iter

input_dim = len(src.vocab)
output_dim = len(trg.vocab)

encoder_in_dim = 256
decoder_in_dim = 256
hidden_dim = 512

n_layers = 2
dropout_ratio = 0.5


# encoder, decoder 객체 선언 
enc = Encoder(input_dim, encoder_in_dim, hidden_dim, n_layers, dropout_ratio)
dec = Decoder(output_dim, decoder_in_dim, hidden_dim, n_layers, dropout_ratio)

# seq2seq 객체 선언
model = Seq2Seq(enc, dec, device).to(device)


# 논문의 내용 대로 (-0.08, 0.08) 값으로 가중치 파라미터 초기화
def init_weights(m):

    for name, param in m.named_parameters():

        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


optimizer = optim.Adam(model.parameters())

# 뒷 부분 패딩에 대해서는 값 무시
trg_pad_idx = trg.vocab.stoi[trg.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

# traing function
def train(model, iterator, optimizer, criterion, clip):

    # train mode
    model.train()

    train_loss = 0
    
    for idx, batch in enumerate(iterator):

        # x_train = [24,128] , label = [31, 128]
        x_train = batch.src
        label = batch.trg

        # 매 iteration 마다 optim 초기화 
        optimizer.zero_grad()

        # output = prediction_string = [target_len, batch_size, len(trg.vocab)] = [30(iter 마다 랜덤), 128, 5893]
        output = model(x_train, label)
        #print("train output dim", output.shape[0], output.shape[1], output.shape[2] )
        
        # output 이 3D tensor 이며, [-1] 은 마지막 차원을 의미함으로 , output_dim = [5893]
        output_dim = output.shape[-1]
        #print("output_dim in train", output_dim)

        # output = [ (30-1) * 128, 5893] = [ 3712, 5893 ] = [(단어의 수 -1) * batch_size, output_dim ]
        output = output[1:].view(-1, output_dim)
        #print("output[1:] dim ", output.shape[0], output.shape[1])

        # label_dim = [3712]
        label = label[1:].view(-1)

        # [3712, 5893] => 5893 개의 단어 집합 중 가장 확률이 높은 단어를 계산하는 행위를 단어의 개수만큼(3712) 비교하여 loss 계산
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
            output = model(x_eval, label, 0)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)

            label = label[1:].view(-1)

            loss = criterion(output, label)

            val_loss += loss.item()

    return val_loss / len(iterator)




# training

import math 

n_epoch, clip = 20, 1

best_val_loss = float('inf')


for epoch in range(n_epoch):

    train_loss = train(model, train_iterator, optimizer, criterion, clip)

    val_loss = evaluate(model, val_iterator, criterion)

    if val_loss < best_val_loss :

        best_val_loss = val_loss
        torch.save(model.state_dict(), "seq2seq.pt")
    
    print(f" < epoch : {epoch+1} > ")
    print(f' Train Loss : {train_loss:.3f} | PerPlexity : {math.exp(train_loss):.3f}')
    print(f' Val Loss : {val_loss:.3f} | Val Perplexity : {math.exp(val_loss):.3f}')


# testing
model.load_state_dict(torch.load('seq2seq.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f"Test loss : {test_loss:.3f} | Test PPL : {math.exp(test_loss): .3f}")

