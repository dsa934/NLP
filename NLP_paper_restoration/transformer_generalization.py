# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-09-15


< 복원한 Transformer 모델을 이용한 기계 번역 >

 - 학습한 모델을 사용하여 임의의 입력 데이터에 대한 출력 데이터 확인해보기 


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

    # model 선언시 to(device) 주의 
    model = Transformer(encoder, decoder, train_pad_idx, label_pad_idx, device).to(device)

    model.load_state_dict(torch.load('translation_transformer.pt'))


    return model

def translation(_input ,field_src, field_trg):

    print("=============================")
    print("< Transformer model을 이용한 불어-> 영어 번역 >")
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
    ex_tensor = torch.LongTensor(ex_index).unsqueeze(0).to(device)
    print(f"입력 텐서 차원 : {ex_tensor.shape}")


    # model 구축
    model = model_set_up()

    # train_mask 생성
    train_mask = model.make_train_mask(ex_tensor)

    # 평가모드
    model.eval()

    # context vector 계산
    with torch.no_grad():

        # encoder_output = []
        encoder_output = model.encoder(ex_tensor, train_mask)
        
    # decoder 입력 단어 정수 임베딩 , 시작은 <sos>
    ex_decoder_index  = [field_trg.vocab.stoi[field_trg.init_token]]

    # <eos> token을 만나면 종료, 
    # 예제의 단어 수가 100개 미만 이라고 가정 
    for time_stemp in range(100):

        # ex_decoder_index[-1] : elements
        # [ex_decoder_index[-1]] : torch를 통한 tensor화는 list를 요소로 사용 함으로 elements를 list화 
        ex_decoder_tensor = torch.LongTensor( ex_decoder_index ).unsqueeze(0).to(device)

        label_mask = model.make_label_mask(ex_decoder_tensor)

        with torch.no_grad():

            # decoder_output = [1,6, 5893], attention = [1,8,6,6]
            decoder_output, attention = model.decoder(ex_decoder_tensor, encoder_output,  label_mask, train_mask )
            
            print(f"general decoder : {decoder_output.shape} , attention : {attention.shape}")
        pred_token = decoder_output.argmax(2)[:,-1].item()

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
