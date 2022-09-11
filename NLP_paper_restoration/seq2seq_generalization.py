# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-05


< ������ Seq2Seq ���� �̿��� ��� ���� >

 - �н��� ���� ����Ͽ� ������ �Է� �����Ϳ� ���� ��� ������ Ȯ���غ��� 


   cf : ������ ���� 

   * Seq2Seq ������ ����ϱ�, �Է� �������� ������ �������� ���, ������ ����� �� ���ٴ� ������ ������, �̿� ���� ���� ����

      �ҽ� ����: ['.', 'freien', 'im', 'tag', 'schonen', 'einen', 'genieben', 'sohn', 'kleiner', 'ihr', 'und', 'mutter', 'eine']
      Ÿ�� ����: ['a', 'mother', 'and', 'her', 'young', 'song', 'enjoying', 'a', 'beautiful', 'day', 'outside', '.']

   1. ���� : �н� �ܰ��� ��ūȭ����, �Է� �������� ���� ������ (o) 

      a) test �ܰ迡�� �Է� �����͸� ���� ������ (o)  : <sos> <unk> in the . <eos>

      b) test �ܰ迡�� �Է� �����͸� ���� ������ (x)  : <sos> a mother and a son enjoying a stroll in a sunny day . <eos>



   2. ���� : �н� �ܰ��� ��ūȭ����, �Է� �������� ���� ������ (x)

      a) test �ܰ迡�� �Է� �����͸� ���� ������ (o)  : <sos> a mother and her daughter are a in a park . <eos>

      b) test �ܰ迡�� �Է� �����͸� ���� ������ (x)  : <sos> a is a a a a child child on the beach. <eos>



   => Ÿ�� ����� ���غ������� ������ ����� ���� best�� ����

      * �Է� �������� ���� ������ (x) in training 

      * �Է� �������� ���� ������ (o) in testing 

      
      
      �� seq2seq paper�� ����� ������ ������ 

      * Multi30k dataset ���  : seq2seq paper���� ����  WMT'14 dataset�� ���� ��������� ���� ���� �����ͼ�

      * 2-layers LSTM : seq2seq paper������ 4-layers LSTM ���

      * �н� epoch ��ü�� ���� �������� ����

      ���� ���� ������ ���̸� ���̴� �� �ϴ�.

 
 
'''

import torch
import torch.nn as nn
import spacy

from seq2seq_model import Encoder, Decoder, Seq2Seq
from seq2seq_preprocessing import src, trg, device , x_train, x_val, y_test


def model_set_up():

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

    model.load_state_dict(torch.load('seq2seq.pt'))

    return model

def translation(_input ,field_src, field_trg):
    
    print("=============================")
    print("< Seq2Seq model�� �̿��� �Ҿ�-> ���� ���� >")
    print("=============================")

    # data input 
    if _input == None  : 

        print("������ ���Ͼ� ���� �Է� : ", end = "")

        _input = input()

        de_tok = spacy.load('de_core_news_sm')

        ex_tokens = [ value.text.lower() for value in de_tok(_input)][::-1]


    else :

        ex_tokens = [ value.lower() for value in _input ][::-1]



    # �Է� seq�� ���Ͽ� <sos> , <eos> �߰� 
    ex_token = [field_src.init_token] + ex_tokens + [field_src.eos_token]
    print(f"��ü �ҽ� ��ū : {ex_token}")

    # encoder�� �Է� �ܾ� ���� �Ӻ���
    ex_index = [ field_src.vocab.stoi[word] for word in ex_token ]
    print(f"�ܾ��� ���� �Ӻ��� : {ex_index}")

    # �Է� ������ �ټ�ȭ
    # ex_tensor = [�ܾ��� ��, 1]
    ex_tensor = torch.LongTensor(ex_index).unsqueeze(1).to(device)
    print(f"�Է� �ټ� ���� : {ex_tensor.shape}")


    # model ����
    model = model_set_up()

    # �򰡸��
    model.eval()

    # context vector ���
    with torch.no_grad():

        # hidden  =[2, 1, 512]
        hidden, cell = model.encoder(ex_tensor)
        
    # decoder �Է� �ܾ� ���� �Ӻ��� , ������ <sos>
    ex_decoder_index  = [field_trg.vocab.stoi[field_trg.init_token]]

    # <eos> token�� ������ ����, 
    # ������ �� ������ 50���� �ܾ�� �̷������ �ʾƼ� ���Ѽ��� 50
    for time_stemp in range(50):

        # ex_decoder_index[-1] : elements
        # [ex_decoder_index[-1]] : torch�� ���� tensorȭ�� list�� ��ҷ� ��� ������ elements�� listȭ 
        ex_decoder_tensor = torch.LongTensor( [ ex_decoder_index[-1] ] ).to(device)

        with torch.no_grad():

            # decoder_output = [1,5893], hidden = [2,1,512]
            decoder_output, hidden, cell = model.decoder(ex_decoder_tensor, hidden, cell )
            
        pred_token = decoder_output.argmax(1).item()

        # �� time step �� ����
        ex_decoder_index.append(pred_token)

        # <eos> ������ ����
        if pred_token == field_trg.vocab.stoi[field_trg.eos_token] : break

    prediction_string = [trg.vocab.itos[index] for index in ex_decoder_index ]

    print(f"model�� ���� ��� : {prediction_string}")




# exmaple 1 
example_idx = 10

de_example = vars(y_test.examples[example_idx])['src']
en_example = vars(y_test.examples[example_idx])['trg']

print(f'�ҽ� ����: {de_example}')
print(f'Ÿ�� ����: {en_example}')

translation( de_example, src, trg)



# example 2 
de_example = None

translation( de_example, src, trg)
