
# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-06


< Seq2Seq with attention model 복원 >

 - Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015 Oral)


'''


from attention_model import Encoder, Decoder, Attention, Seq2Seq
import seq2seq_preprocessing 
import torch
import torch.nn as nn 
import torch.optim as optim

# set hyper params
src, trg, device = seq2seq_preprocessing.src , seq2seq_preprocessing.trg, seq2seq_preprocessing.device
train_iterator , val_iterator , test_iterator = seq2seq_preprocessing.train_iter , seq2seq_preprocessing.val_iter, seq2seq_preprocessing.test_iter

input_dim, output_dim = len(src.vocab), len(trg.vocab)
encoder_embed_dim, decoder_embed_dim = 256, 256
encoder_hidden_dim, decoder_hidden_dim = 512, 512
dropout_ratio = 0.5 

# model setup
attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
encoder = Encoder(input_dim, encoder_embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout_ratio)
decoder = Decoder(output_dim, decoder_embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout_ratio, attention)
model = Seq2Seq(encoder, decoder, device).to(device)

# model params init
def init_weights(m):

    for name, param in m.named_parameters():

        if 'weight' in name : nn.init.normal_(param.data, mean = 0 , std = 0.01)

        else : nn.init.constant_(param.data, 0)

model.apply(init_weights)

# optimizer set
optimizer = optim.Adam(model.parameters())

# ignore pad token 
trg_pad_idx = trg.vocab.stoi[trg.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)


# traing function
def train(model, iterator, optimizer, criterion, clip):

    # train mode
    model.train()

    train_loss = 0
    
    for idx, batch in enumerate(iterator):

        # x_train = [단어의 수(de), batch_size] , label = [단어의 수(en), batch_size]
        x_train = batch.src
        label = batch.trg

        # 매 iteration 마다 optim 초기화 
        optimizer.zero_grad()

        # output = [ target_len, batch_size, len(trg.vocab)] = [30(iter 마다 랜덤), 128, 5893]
        output = model(x_train, label)
        
        # output은 3D tensor 이며, [-1] 은 마지막 차원을 의미함으로 , output_dim = [5893]
        output_dim = output.shape[-1]
        
        # output = [ (30-1) * 128, 5893] = [ 3712, 5893 ] = [(단어의 수 -1) * batch_size, output_dim ]
        output = output[1:].view(-1, output_dim)
        
        label = label[1:].view(-1)
        
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

n_epoch, clip = 10, 1

best_val_loss = float('inf')


for epoch in range(n_epoch):

    train_loss = train(model, train_iterator, optimizer, criterion, clip)

    val_loss = evaluate(model, val_iterator, criterion)

    if val_loss < best_val_loss :

        best_val_loss = val_loss
        torch.save(model.state_dict(), "attention_seq2seq.pt")
    
    print(f" < epoch : {epoch+1} > ")
    print(f' Train Loss : {train_loss:.3f} | PerPlexity : {math.exp(train_loss):.3f}')
    print(f' Val Loss : {val_loss:.3f} | Val Perplexity : {math.exp(val_loss):.3f}')


# testing
model.load_state_dict(torch.load('attention_seq2seq.pt'))
test_loss = evaluate(model, test_iterator, criterion)
print(f"Test loss : {test_loss:.3f} | Test PPL : {math.exp(test_loss): .3f}")

