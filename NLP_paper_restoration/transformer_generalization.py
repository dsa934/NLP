# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-15


< ������ Transformer ���� �̿��� ��� ���� >

 - �н��� ���� ����Ͽ� ������ �Է� �����Ϳ� ���� ��� ������ Ȯ���غ��� 


'''

import torch
import torch.nn as nn
import spacy


from transformer_model import Multi_head_Attention, simple_ff, Encoder , Encoder_layer, Decoder, Decoder_layer, Transformer
from transformer_preprocessing import src, trg, device , x_train, x_val, y_test


def model_set_up():

    input_dim, output_dim = len(src.vocab), len(trg.vocab)
    encoder_embed_dim, decoder_embed_dim = 256, 256
    layers, heads, ff_dim, dropout_ratio = 3, 8, 512, 0.1

    train_pad_idx = src.vocab.stoi[src.pad_token]
    label_pad_idx = trg.vocab.stoi[trg.pad_token]

    # model setup
    encoder = Encoder(input_dim, encoder_embed_dim, layers, heads, ff_dim, dropout_ratio, device)
    decoder = Decoder(output_dim , decoder_embed_dim, layers, heads, ff_dim, dropout_ratio, device)

    # model ����� to(device) ���� 
    model = Transformer(encoder, decoder, train_pad_idx, label_pad_idx, device).to(device)

    model.load_state_dict(torch.load('translation_transformer.pt'))


    return model

def translation(_input ,field_src, field_trg):

    print("=============================")
    print("< Transformer model�� �̿��� �Ҿ�-> ���� ���� >")
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
    ex_tensor = torch.LongTensor(ex_index).unsqueeze(0).to(device)
    print(f"�Է� �ټ� ���� : {ex_tensor.shape}")


    # model ����
    model = model_set_up()

    # train_mask ����
    train_mask = model.make_train_mask(ex_tensor)

    # �򰡸��
    model.eval()

    # context vector ���
    with torch.no_grad():

        # encoder_output = []
        encoder_output = model.encoder(ex_tensor, train_mask)
        
    # decoder �Է� �ܾ� ���� �Ӻ��� , ������ <sos>
    ex_decoder_index  = [field_trg.vocab.stoi[field_trg.init_token]]

    # <eos> token�� ������ ����, 
    # ������ �ܾ� ���� 100�� �̸� �̶�� ���� 
    for time_stemp in range(100):

        # ex_decoder_index[-1] : elements
        # [ex_decoder_index[-1]] : torch�� ���� tensorȭ�� list�� ��ҷ� ��� ������ elements�� listȭ 
        ex_decoder_tensor = torch.LongTensor( ex_decoder_index ).unsqueeze(0).to(device)

        label_mask = model.make_label_mask(ex_decoder_tensor)

        with torch.no_grad():

            # decoder_output = [1,6, 5893], attention = [1,8,6,6]
            decoder_output, attention = model.decoder(ex_decoder_tensor, encoder_output,  label_mask, train_mask )
            
            print(f"general decoder : {decoder_output.shape} , attention : {attention.shape}")
        pred_token = decoder_output.argmax(2)[:,-1].item()

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
