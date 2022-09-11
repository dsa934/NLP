# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< IMDB 리뷰 감성 분류하기 < M:1 RNN을 이용한  >

   * IMDB 리뷰 데이터 
   
      - ML에서 텍스트 분류 연습을 위해 자주 사용 되는 데이터

      - 2개의 column 으로 구성 ( review , sentiment(긍정/부정 판단) )

      - total sample : 50,000

        Sample ratio ( train / test ) = 392 : 391  

        sample ratio (train / val( train's 20%) ) = 313 : 79 




<Step>

1. 데이터 전처리

   1-1. data field 전처리 

        => 2개의 field(text, label) 객체 정의

             실제 훈련 데이터를 의미하는 것이 아니라,  해당 field의 형태에 맞게 처리하겠다는 의미 


   1-2. 데이터를 field 객체에 맞게 토큰화

        => 데이터를 로드하여, 1-1 에서 선언한 field 에 맞개 토큰화(분할)


   1-3 : 단어 집합 형성

        => 토큰화 이후 정수 인코딩 진행 , 5번 미만 으로 등장한 단어는 <unk> 로 대체


   1-4 : 데이터 로더 구축

        => 데이터 셋으로부터 mini-batch만큼 데이터를 로드하게 만들어주는 역할 
           
           보통 iterator 사용하지만, 해당 예제에서는 BucketIterator을 사용 





2. RNN 모델 구현 

   2-1. model 생성 및 학습(train), 평가(evaluate) 함수 생성



   2-2. 학습 with training data

       => Model의 forward propagation 과정에서의 차원 변화 

          h_t = [64,256]

          init_data -> after embedding layer -> after GRU layer  ->  output 

          [ 64,950 ] ->  [ 64, 950, 128 ]    -> [ 64, 950, 256 ] ->  [ 64, 2 ]


          [64, 950] = [batch_size, review_length]
          
           => 64개의 review 데이터, 64개의 review 중 가장 긴 review의 value로 setting ( <pad> 토큰을 이용해 부족한 경우 padding )

              review 길이의 제한이 없음으로, 각 batch 마다 review 길이가 다를 수 있다.



          [64, 950, 128]

           => embedding layer의 입출력 = [vocab_size, embedding_dim]

             * 단어 집합

               => review 데이터에 존재하는 모든 단어 중 빈도수가 5회 이상인 데이터로 단어 집합을 구성하여 각 단어별 index를 부여했기 때문에

                  embedding layer의 입력으로 사용되는 vector는 단어집합의 크기 만큼의 차원을 갖는 vector가 된다.

              
              * [64, 128] 이아니라 [64, 950, 128]인 이유

                => 해당 NLP Task 는 Many to one 임으로 950(seq_len)에 대한 긍정/부정 예측값 1개를 출력하는 형태

                   [1,1,128] : 첫번째 리뷰의 첫번쨰 단어를 128 차원으로 임베딩 한다는 의미가 됨으로 부적합

                   [1,950,128] : 첫번쨰 리뷰(첫번쨰 리뷰의 모든 길이, 문장에 대해) 128 차원으로 임베딩 한다는 의미가 됨으로 적합

                   즉, 더이상 review_sequence 가 아닌 interger sequence 


          [64, 950, 256]

           => GRU layer의 입출력 = [embedding_dim, hidden_dim] 임으로 ,

              128 차원의 embedding vector가 gru을 통해 256 차원의 hidden state vector 로 변환 

              128, 256 은 사용자 정의 값 


          [64, 2]

           => 해당 NLP Task는 many to one 에 해당하는 텍스트 분류(긍정,부정)임으로 출력값은 2가 되야 함 



       => h_t = x[:,-1,:] 이 갖는 의미 

          syntax example)  x.shape = [3,4,5]  =>  x[:,-1,:] =>  x.shape = [3,5]

          GRU layer를 통과하면 상술하였듯, [batch, seq_len, hidden_dim]의 형태로 결과값이 도출되지만

          GRU cell 연산 관점에서 seq_len은 불필요한 정보임으로 차원을 줄임으로써 
          
          다음 hidden state 연산에 필요한 [batch, hidden_dim] 값만 처리함으로써 연산 시간을 단축 시킴 


          
   2-3. 평가 with validation data

   2-4. 테스트 with test_data

       => evaluate 함수의 경우 (validation, test) 와 같이 데이터의 종류만 다를 뿐 기능 자체는 동일

         evaluate(validation_data)을 통해 overfitting check

         evaluate(test_data)를 통해 모델의 performance check

          



* torch.cuda.is_available() => False 일 경우,

  => pytorch, cuda, cudnn의 버전이 맞지 않는 경우 학습에 GPU를 사용할 수 없다 

     nvidia-smi 명령어를 통해 GPU, Cuda version을 확인한 후

     pytorch, cuda, cudnn 의 각 버전이 맞게 설치 

     * 위 3개의 버전이 맞아도 torchtext.legacy가 작동이 안되는 경우가 있어, legacy 까지 고려한 버전 맞춤이 필요


     ∴ # CUDA 11.1
        pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html



'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchtext 의 data, datasets 은 torchtext.legacy로 옮겨짐
# pip install pytorch==0.9를 통해 버전을 변경해야 legacy 속성에 접근 가능
# 일반적인 pytorch 설치 시 해당 기능이 작동하지 않음 

from torchtext.data import Field, BucketIterator
import torchtext.datasets 
#from torchtext.legacy import data, datasets

import random

print("chk",torch.cuda.is_available())

# set random seed
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)

# hyperparameters
batch_size = 64
learning_rate = 0.001
epoch = 10

# GPU Settings
use_cuda = torch.cuda.is_available()
device = torch.device('cuda')

# Step 1 : 데이터 전처리


# 1-1. data field 전처리 
#sequential : 시퀀스 데이터 여부. (True가 기본값)
#use_vocab : 단어 집합을 만들 것인지 여부. (True가 기본값)
#tokenize : 어떤 토큰화 함수를 사용할 것인지 지정. (string.split이 기본값)
#lower : 영어 데이터를 전부 소문자화한다. (False가 기본값)
#batch_first : 미니 배치 차원을 맨 앞으로 하여 데이터를 불러올 것인지 여부. (False가 기본값)
#is_target : 레이블 데이터 여부. (False가 기본값)
#fix_length : 최대 허용 길이. 이 길이에 맞춰서 패딩 작업(Padding)이 진행된다.

text = data.Field(sequential = True, batch_first = True, lower = True)
label = data.Field(sequential = False, batch_first = True)

# 1-2 : 데이터를 field 객체에 맞게 토큰화
x_train, y_test = datasets.IMDB.splits(text, label)

# 첫번쨰 훈련 데이터 샘플 확인
print(vars(x_train[0]))

# 1-3 : 단어 집합 형성
text.build_vocab(x_train, min_freq=5)
label.build_vocab(x_train)

vocab_size = len(text.vocab)

# 부정 /긍정 만 구분하면됨
n_classes = 2

# stoi 를 통해 각 단어에 대한 정수 인덱스 확인 가능
print(text.vocab.stoi)


# 1-4 : 데이터 로더 구축
x_train, x_val = x_train.split(split_ratio = 0.8)

# BucketIterator : 모든 단어를 index 번호로 대체하는 기능
# 각 데이터 별 mini-batch 수 
# train : 313, val :79, test : 391
train_iter, val_iter, test_iter = data.BucketIterator.splits((x_train, x_val, y_test), batch_size=batch_size, shuffle=True, repeat=False)



# Step 2 : RNN 모델 구축

# 2-1. model 생성 및 학습(train), 평가(evaluate) 함수 생성

class JW_GRU(nn.Module):

    # model의 init은 신경망 구성을 나타내는 것
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes, n_layers, dropout_ratio = 0.2 ):

        super(JW_GRU, self).__init__()

        # 최초 입력은 단어 집합의 크기 차원을 갖는 벡터 -> 임베딩 레이어 통과 후 임베딩 백터 사이즈로 변환
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout_ratio)

        self.n_layers= n_layers

        # RNN 계열 모델의 입력은 embedding vector로 시작
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True )

        # 출력부
        self.out = nn.Linear(self.hidden_dim, n_classes)

    # rnn 계열 학습에는 초기값 h_0가 필요
    def _init_state(self, batch_size=1):

        # iter : 반복 가능한 객체를 만듬 
        # next : 반복 가능 객체의 다음 요소 반환 
        # iterations = iter[["html", "CSS","JS"]]
        # next(iterations) : html
        # next(iterations) : css
        weight = next(self.parameters()).data

        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()


    def forward(self, x):

        # init data shape = [64,950] 

        # after embedding layer = [64,950,128]        
        x = self.embedding(x)
        
        h_0 = self._init_state(batch_size = x.size(0))
        
        # after GRU layer = [64, 950, 256]
        # many to one 구조임으로 GRU 연산은 초기값만 셋팅해주면 됨 
        # many to many인 경우 초기값, 초기값이 아닌 경우를 나눠서
        # GRU(x, h_0), GRU(x,h_t)와 같은 형태로 진행 해야 함 
        x, _= self.gru(x, h_0)

        #h_t_shape = [64, 256]
        h_t = x[:,-1,:]
        
        self.dropout(h_t)

        #output_shape = [64,2]
        logit = self.out(h_t)
        
        return logit


# 모델 생성 및 CUDA로 학습
GRU_model = JW_GRU(vocab_size,128, 256, n_classes, 1, 0.5).to(device)
optimizer = optim.Adam(GRU_model.parameters(), lr= learning_rate)


# set train fucntion
def train(GRU_model, optimizer, train_iter):
    
    # train 선언
    GRU_model.train()

    for _, batch in enumerate(train_iter):

        # 이 구간이 실제 훈련 데이터에 text, label field를 적용 하는 부분 
        x, y = batch.text.to(device), batch.label.to(device)
    
        # y.data.sub_(1) : y.data 에서 1을 뺸다 
        # 현재 y.data tensor는 1,2로 구성되어 있음 
        # 레이블 값을 0, 1 로 변환
        y.data.sub_(1)

        # 매 batch 마다 optimizer 초기화 필요 ( 각 batch 마다 다른 데이터 분포가 input data로 활용 )
        optimizer.zero_grad()

        logit = GRU_model(x)

        loss = F.cross_entropy(logit, y)

        loss.backward()

        optimizer.step()


# set evaluate function
def evaluate(GRU_model, val_iter):
    
    print("model evaluating")

    GRU_model.eval()
    
    correct, total_loss = 0, 0 

    for batch in val_iter:

        x, y = batch.text.to(device), batch.label.to(device)

        y.data.sub_(1)

        logit = GRU_model(x)

        # 정답, 예측값에 대한 cross entropy 값들의 합으로 return ( default : mean )
        # 해당 예제에서는 mean, sum 방법에 따른 차이는 미비
        loss = F.cross_entropy(logit, y, reduction='sum')

        # 계속 loss를 누적해야 평균 loss 계산 가능 
        total_loss += loss.item()

        correct += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)

    avg_loss = total_loss / size

    avg_accuracy = 100.0 * correct / size

    return avg_loss, avg_accuracy


# 2-2. 학습 with training data
best_val_loss = None

for _epoch in range(1, epoch+1):

    train(GRU_model, optimizer, train_iter)

    # 2-3. 평가 with validation data
    val_loss, val_accuracy = evaluate(GRU_model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (_epoch, val_loss, val_accuracy))

    # min(val_loss) 일떄의 모델 저장
    if not best_val_loss or val_loss < best_val_loss:

        if not os.path.isdir("snapshot"):

            os.makedirs("snapshot")

        torch.save(GRU_model.state_dict(), './snapshot/txtclassification.pt')

        best_val_loss = val_loss


# testing

# load best performance models
GRU_model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))

# 2-4. 테스트 with test_data
test_loss, test_acc = evaluate(GRU_model, test_iter)

# show result
print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))