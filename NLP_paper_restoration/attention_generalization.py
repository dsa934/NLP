# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-07


< 복원한 Attention 모델을 이용한 기계 번역 >

 - 학습한 모델을 사용하여 임의의 입력 데이터에 대한 출력 데이터 확인해보기 


   cf : 구현시 참고 

   * Seq2Seq_generalization.py 의 내용 참조 


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
    print("< Attention model을 이용한 불어-> 영어 번역 >")
    print("=============================")

    # data input 
    if _input == None  : 

        print("번역할 독일어 문장 입력 : ", end = "")

        _input = input()

        de_tok = spacy.load('de_core_news_sm')

        ex_tokens = [ value.text.lower() for value in de_tok(_input)][::-1]


    else :

        ex_tokens = [ value.lower() for value in _input ][::-1]



    # 입력 seq에 대하여 <sos> , <eos> 추가 
    ex_token = [field_src.init_token] + ex_tokens + [field_src.eos_token]
    print(f"전체 소스 토큰 : {ex_token}")

    # encoder의 입력 단어 정수 임베딩
    ex_index = [ field_src.vocab.stoi[word] for word in ex_token ]
    print(f"단어의 정수 임베딩 : {ex_index}")

    # 입력 데이터 텐서화
    # ex_tensor = [단어의 수, 1]
    ex_tensor = torch.LongTensor(ex_index).unsqueeze(1).to(device)
    print(f"입력 텐서 차원 : {ex_tensor.shape}")


    # model 구축
    model = model_set_up()

    # 평가모드
    model.eval()

    # context vector 계산
    with torch.no_grad():

        # encoder_output = [7,1,1024] & hidden  = [1,512]
        encoder_output, hidden = model.encoder(ex_tensor)
        
    # decoder 입력 단어 정수 임베딩 , 시작은 <sos>
    ex_decoder_index  = [field_trg.vocab.stoi[field_trg.init_token]]

    # <eos> token을 만나면 종료, 
    # 예제는 한 문장이 50개의 단어로 이루어지진 않아서 상한선을 50
    for time_stemp in range(50):

        # ex_decoder_index[-1] : elements
        # [ex_decoder_index[-1]] : torch를 통한 tensor화는 list를 요소로 사용 함으로 elements를 list화 
        ex_decoder_tensor = torch.LongTensor( [ ex_decoder_index[-1] ] ).to(device)

        with torch.no_grad():

            # decoder_output = [], hidden = []
            decoder_output, hidden = model.decoder(ex_decoder_tensor, hidden, encoder_output )
            
        pred_token = decoder_output.argmax(1).item()

        # 매 time step 별 갱신
        ex_decoder_index.append(pred_token)

        # <eos> 만나면 종료
        if pred_token == field_trg.vocab.stoi[field_trg.eos_token] : break

    prediction_string = [trg.vocab.itos[index] for index in ex_decoder_index ]

    print(f"model의 예측 결과 : {prediction_string}")




# exmaple 1 
example_idx = 10

de_example = vars(y_test.examples[example_idx])['src']
en_example = vars(y_test.examples[example_idx])['trg']

print(f'소스 문장: {de_example}')
print(f'타겟 문장: {en_example}')

translation( de_example, src, trg)



# example 2 
de_example = None

translation( de_example, src, trg)
