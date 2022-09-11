# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-05


< 복원한 Seq2Seq 모델을 이용한 기계 번역 >

 - 학습한 모델을 사용하여 임의의 입력 데이터에 대한 출력 데이터 확인해보기 


   cf : 구현시 참고 

   * Seq2Seq 논문에서 언급하길, 입력 데이터의 순서를 뒤집었을 경우, 번역의 결과가 더 좋다는 내용이 있으며, 이에 대한 실험 진행

      소스 문장: ['.', 'freien', 'im', 'tag', 'schonen', 'einen', 'genieben', 'sohn', 'kleiner', 'ihr', 'und', 'mutter', 'eine']
      타겟 문장: ['a', 'mother', 'and', 'her', 'young', 'song', 'enjoying', 'a', 'beautiful', 'day', 'outside', '.']

   1. 조건 : 학습 단계의 토큰화에서, 입력 데이터의 순서 뒤집기 (o) 

      a) test 단계에서 입력 데이터를 순서 뒤집기 (o)  : <sos> <unk> in the . <eos>

      b) test 단계에서 입력 데이터를 순서 뒤집기 (x)  : <sos> a mother and a son enjoying a stroll in a sunny day . <eos>



   2. 조건 : 학습 단계의 토큰화에서, 입력 데이터의 순서 뒤집기 (x)

      a) test 단계에서 입력 데이터를 순서 뒤집기 (o)  : <sos> a mother and her daughter are a in a park . <eos>

      b) test 단계에서 입력 데이터를 순서 뒤집기 (x)  : <sos> a is a a a a child child on the beach. <eos>



   => 타겟 문장과 비교해보았을떄 번역의 결과가 가장 best인 경우는

      * 입력 데이터의 순서 뒤집기 (x) in training 

      * 입력 데이터의 순서 뒤집기 (o) in testing 

      
      
      ∴ seq2seq paper의 내용과 상이한 이유는 

      * Multi30k dataset 사용  : seq2seq paper에서 사용된  WMT'14 dataset에 비해 상대적으로 적은 양의 데이터셋

      * 2-layers LSTM : seq2seq paper에서는 4-layers LSTM 사용

      * 학습 epoch 자체가 많게 설정되지 않음

      위와 같은 이유로 차이를 보이는 듯 하다.

 
 
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


    # encoder, decoder 객체 선언 
    enc = Encoder(input_dim, encoder_in_dim, hidden_dim, n_layers, dropout_ratio)
    dec = Decoder(output_dim, decoder_in_dim, hidden_dim, n_layers, dropout_ratio)

    # seq2seq 객체 선언
    model = Seq2Seq(enc, dec, device).to(device)

    model.load_state_dict(torch.load('seq2seq.pt'))

    return model

def translation(_input ,field_src, field_trg):
    
    print("=============================")
    print("< Seq2Seq model을 이용한 불어-> 영어 번역 >")
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

        # hidden  =[2, 1, 512]
        hidden, cell = model.encoder(ex_tensor)
        
    # decoder 입력 단어 정수 임베딩 , 시작은 <sos>
    ex_decoder_index  = [field_trg.vocab.stoi[field_trg.init_token]]

    # <eos> token을 만나면 종료, 
    # 예제는 한 문장이 50개의 단어로 이루어지진 않아서 상한선을 50
    for time_stemp in range(50):

        # ex_decoder_index[-1] : elements
        # [ex_decoder_index[-1]] : torch를 통한 tensor화는 list를 요소로 사용 함으로 elements를 list화 
        ex_decoder_tensor = torch.LongTensor( [ ex_decoder_index[-1] ] ).to(device)

        with torch.no_grad():

            # decoder_output = [1,5893], hidden = [2,1,512]
            decoder_output, hidden, cell = model.decoder(ex_decoder_tensor, hidden, cell )
            
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
