# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-03


< ����ȭ Seq2Seq model �����ϱ� >

 - Sequence to Sequence Learning with Neural Networks (NIPS 2014) 

 - ������ �̽� (���� ��ǻ���� ���� �Ѱ�)�� ���� ���� Seq2Seq���� ���� <> dataset ��� �뷮�� ��������� ����  Multi30k �����ͼ��� Ȱ���Ͽ�

   seq2seq model �����غ��� 


<Step>

1. ������ ��ó��

   - seq2seq_preprocessing.py


2. �� ����

   - seq2seq_model.py


3. Seq2Seq ����

   1. encoder, decoder, seq2seq model ��ü ���� 

   2. Parameter �ʱ�ȭ ( ������ ��� )

   3. train ,evaluate function set 

   4. training, testing



  cf : ������ ���� 

   * ��ūȭ �� �����Ϳ� ���Ͽ� �е� �����ʹ� ���� �ϱ� 


   * train, label �������� �ܾ� �� 

     => �Ҿ�� ����� �����ϴ� task ������, 
    
        �Է����� ���� train data�� �ܾ� �� ��  �������� Ȱ��Ǵ� label�� �ܾ� ���� �ٸ� �� �ִ�.


   * clip 

     => RNN �迭 �Ű�������� Gradient Vanishing or exploding�� ���� �߻��ϴµ� �̸� �����ϱ� ���� ���

        ���� �Ӱ谪(Threshold)�� �ʰ��� ��� clip(�ڸ���, ����)


   * CrossEntropy

     =>  output_dim = [ 3712, 5893 ]  : (30-1) * 128  , 30���� �ܾ� �� <sos> �� �����ϰ�, 128���� batch_size�� �����ϴ� �ܾ ��� ����

         label_dim = [ 3712 ]

         => 3712 ���� �ܾ ���Ͽ� , 5893���� �ܾ� �ĺ� �� �� ���� Ȯ���� ���� �ܾ ã�� ���� CrossEntropy ���� 




'''

import seq2seq_preprocessing 
from seq2seq_model import Encoder, Decoder, Seq2Seq
import torch
import torch.nn as nn 
import torch.optim as optim

# init hyperparams
# src, trg �� multi30k �����Ϳ��� �̹� field column ������ ���� �ؼ� ������ �Ұ���
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


# encoder, decoder ��ü ���� 
enc = Encoder(input_dim, encoder_in_dim, hidden_dim, n_layers, dropout_ratio)
dec = Decoder(output_dim, decoder_in_dim, hidden_dim, n_layers, dropout_ratio)

# seq2seq ��ü ����
model = Seq2Seq(enc, dec, device).to(device)


# ���� ���� ��� (-0.08, 0.08) ������ ����ġ �Ķ���� �ʱ�ȭ
def init_weights(m):

    for name, param in m.named_parameters():

        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)


optimizer = optim.Adam(model.parameters())

# �� �κ� �е��� ���ؼ��� �� ����
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

        # �� iteration ���� optim �ʱ�ȭ 
        optimizer.zero_grad()

        # output = prediction_string = [target_len, batch_size, len(trg.vocab)] = [30(iter ���� ����), 128, 5893]
        output = model(x_train, label)
        #print("train output dim", output.shape[0], output.shape[1], output.shape[2] )
        
        # output �� 3D tensor �̸�, [-1] �� ������ ������ �ǹ������� , output_dim = [5893]
        output_dim = output.shape[-1]
        #print("output_dim in train", output_dim)

        # output = [ (30-1) * 128, 5893] = [ 3712, 5893 ] = [(�ܾ��� �� -1) * batch_size, output_dim ]
        output = output[1:].view(-1, output_dim)
        #print("output[1:] dim ", output.shape[0], output.shape[1])

        # label_dim = [3712]
        label = label[1:].view(-1)

        # [3712, 5893] => 5893 ���� �ܾ� ���� �� ���� Ȯ���� ���� �ܾ ����ϴ� ������ �ܾ��� ������ŭ(3712) ���Ͽ� loss ���
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

