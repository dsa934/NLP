# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-07


< ������ Attention ���� �̿��� ��� ���� >

 - �н��� ���� ����Ͽ� ������ �Է� �����Ϳ� ���� ��� ������ Ȯ���غ��� 


   cf : ������ ���� 

   * Seq2Seq_generalization.py �� ���� ���� 


'''

import torch
import torch.nn as nn
import spacy

from attention_model import Encoder, Decoder, Seq2Seq, Attention
from seq2seq_preprocessing import src, trg, device , x_train, x_val, y_test


def model_set_up():

    input_dim, output_dim = len(src.vocab), len(trg.vocab)
    encoder_embed_dim, decoder_embed_dim = 256, 256
    encoder_hidden_dim, decoder_hidden_dim = 512, 512
    dropout_ratio = 0.5 

    # model setup
    attention = Attention(encoder_hidden_dim, decoder_hidden_dim)
    encoder = Encoder(input_dim, encoder_embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout_ratio)
    decoder = Decoder(output_dim, decoder_embed_dim, encoder_hidden_dim, decoder_hidden_dim, dropout_ratio, attention)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(torch.load('attention_seq2seq.pt'))

    return model

def translation(_input ,field_src, field_trg):

    print("=============================")
    print("< Attention model�� �̿��� �Ҿ�-> ���� ���� >")
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

        # encoder_output = [7,1,1024] & hidden  = [1,512]
        encoder_output, hidden = model.encoder(ex_tensor)
        
    # decoder �Է� �ܾ� ���� �Ӻ��� , ������ <sos>
    ex_decoder_index  = [field_trg.vocab.stoi[field_trg.init_token]]

    # <eos> token�� ������ ����, 
    # ������ �� ������ 50���� �ܾ�� �̷������ �ʾƼ� ���Ѽ��� 50
    for time_stemp in range(50):

        # ex_decoder_index[-1] : elements
        # [ex_decoder_index[-1]] : torch�� ���� tensorȭ�� list�� ��ҷ� ��� ������ elements�� listȭ 
        ex_decoder_tensor = torch.LongTensor( [ ex_decoder_index[-1] ] ).to(device)

        with torch.no_grad():

            # decoder_output = [], hidden = []
            decoder_output, hidden = model.decoder(ex_decoder_tensor, hidden, encoder_output )
            
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
