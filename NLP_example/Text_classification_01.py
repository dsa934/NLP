# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< IMDB ���� ���� �з��ϱ� < M:1 RNN�� �̿���  >

   * IMDB ���� ������ 
   
      - ML���� �ؽ�Ʈ �з� ������ ���� ���� ��� �Ǵ� ������

      - 2���� column ���� ���� ( review , sentiment(����/���� �Ǵ�) )

      - total sample : 50,000

        Sample ratio ( train / test ) = 392 : 391  

        sample ratio (train / val( train's 20%) ) = 313 : 79 




<Step>

1. ������ ��ó��

   1-1. data field ��ó�� 

        => 2���� field(text, label) ��ü ����

             ���� �Ʒ� �����͸� �ǹ��ϴ� ���� �ƴ϶�,  �ش� field�� ���¿� �°� ó���ϰڴٴ� �ǹ� 


   1-2. �����͸� field ��ü�� �°� ��ūȭ

        => �����͸� �ε��Ͽ�, 1-1 ���� ������ field �� �°� ��ūȭ(����)


   1-3 : �ܾ� ���� ����

        => ��ūȭ ���� ���� ���ڵ� ���� , 5�� �̸� ���� ������ �ܾ�� <unk> �� ��ü


   1-4 : ������ �δ� ����

        => ������ �����κ��� mini-batch��ŭ �����͸� �ε��ϰ� ������ִ� ���� 
           
           ���� iterator ���������, �ش� ���������� BucketIterator�� ��� 





2. RNN �� ���� 

   2-1. model ���� �� �н�(train), ��(evaluate) �Լ� ����



   2-2. �н� with training data

       => Model�� forward propagation ���������� ���� ��ȭ 

          h_t = [64,256]

          init_data -> after embedding layer -> after GRU layer  ->  output 

          [ 64,950 ] ->  [ 64, 950, 128 ]    -> [ 64, 950, 256 ] ->  [ 64, 2 ]


          [64, 950] = [batch_size, review_length]
          
           => 64���� review ������, 64���� review �� ���� �� review�� value�� setting ( <pad> ��ū�� �̿��� ������ ��� padding )

              review ������ ������ ��������, �� batch ���� review ���̰� �ٸ� �� �ִ�.



          [64, 950, 128]

           => embedding layer�� ����� = [vocab_size, embedding_dim]

             * �ܾ� ����

               => review �����Ϳ� �����ϴ� ��� �ܾ� �� �󵵼��� 5ȸ �̻��� �����ͷ� �ܾ� ������ �����Ͽ� �� �ܾ index�� �ο��߱� ������

                  embedding layer�� �Է����� ���Ǵ� vector�� �ܾ������� ũ�� ��ŭ�� ������ ���� vector�� �ȴ�.

              
              * [64, 128] �̾ƴ϶� [64, 950, 128]�� ����

                => �ش� NLP Task �� Many to one ������ 950(seq_len)�� ���� ����/���� ������ 1���� ����ϴ� ����

                   [1,1,128] : ù��° ������ ù���� �ܾ 128 �������� �Ӻ��� �Ѵٴ� �ǹ̰� ������ ������

                   [1,950,128] : ù���� ����(ù���� ������ ��� ����, ���忡 ����) 128 �������� �Ӻ��� �Ѵٴ� �ǹ̰� ������ ����

                   ��, ���̻� review_sequence �� �ƴ� interger sequence 


          [64, 950, 256]

           => GRU layer�� ����� = [embedding_dim, hidden_dim] ������ ,

              128 ������ embedding vector�� gru�� ���� 256 ������ hidden state vector �� ��ȯ 

              128, 256 �� ����� ���� �� 


          [64, 2]

           => �ش� NLP Task�� many to one �� �ش��ϴ� �ؽ�Ʈ �з�(����,����)������ ��°��� 2�� �Ǿ� �� 



       => h_t = x[:,-1,:] �� ���� �ǹ� 

          syntax example)  x.shape = [3,4,5]  =>  x[:,-1,:] =>  x.shape = [3,5]

          GRU layer�� ����ϸ� ����Ͽ���, [batch, seq_len, hidden_dim]�� ���·� ������� ���������

          GRU cell ���� �������� seq_len�� ���ʿ��� ���������� ������ �������ν� 
          
          ���� hidden state ���꿡 �ʿ��� [batch, hidden_dim] ���� ó�������ν� ���� �ð��� ���� ��Ŵ 


          
   2-3. �� with validation data

   2-4. �׽�Ʈ with test_data

       => evaluate �Լ��� ��� (validation, test) �� ���� �������� ������ �ٸ� �� ��� ��ü�� ����

         evaluate(validation_data)�� ���� overfitting check

         evaluate(test_data)�� ���� ���� performance check

          



* torch.cuda.is_available() => False �� ���,

  => pytorch, cuda, cudnn�� ������ ���� �ʴ� ��� �н��� GPU�� ����� �� ���� 

     nvidia-smi ��ɾ ���� GPU, Cuda version�� Ȯ���� ��

     pytorch, cuda, cudnn �� �� ������ �°� ��ġ 

     * �� 3���� ������ �¾Ƶ� torchtext.legacy�� �۵��� �ȵǴ� ��찡 �־�, legacy ���� ����� ���� ������ �ʿ�


     �� # CUDA 11.1
        pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html



'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchtext �� data, datasets �� torchtext.legacy�� �Ű���
# pip install pytorch==0.9�� ���� ������ �����ؾ� legacy �Ӽ��� ���� ����
# �Ϲ����� pytorch ��ġ �� �ش� ����� �۵����� ���� 

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

# Step 1 : ������ ��ó��


# 1-1. data field ��ó�� 
#sequential : ������ ������ ����. (True�� �⺻��)
#use_vocab : �ܾ� ������ ���� ������ ����. (True�� �⺻��)
#tokenize : � ��ūȭ �Լ��� ����� ������ ����. (string.split�� �⺻��)
#lower : ���� �����͸� ���� �ҹ���ȭ�Ѵ�. (False�� �⺻��)
#batch_first : �̴� ��ġ ������ �� ������ �Ͽ� �����͸� �ҷ��� ������ ����. (False�� �⺻��)
#is_target : ���̺� ������ ����. (False�� �⺻��)
#fix_length : �ִ� ��� ����. �� ���̿� ���缭 �е� �۾�(Padding)�� ����ȴ�.

text = data.Field(sequential = True, batch_first = True, lower = True)
label = data.Field(sequential = False, batch_first = True)

# 1-2 : �����͸� field ��ü�� �°� ��ūȭ
x_train, y_test = datasets.IMDB.splits(text, label)

# ù���� �Ʒ� ������ ���� Ȯ��
print(vars(x_train[0]))

# 1-3 : �ܾ� ���� ����
text.build_vocab(x_train, min_freq=5)
label.build_vocab(x_train)

vocab_size = len(text.vocab)

# ���� /���� �� �����ϸ��
n_classes = 2

# stoi �� ���� �� �ܾ ���� ���� �ε��� Ȯ�� ����
print(text.vocab.stoi)


# 1-4 : ������ �δ� ����
x_train, x_val = x_train.split(split_ratio = 0.8)

# BucketIterator : ��� �ܾ index ��ȣ�� ��ü�ϴ� ���
# �� ������ �� mini-batch �� 
# train : 313, val :79, test : 391
train_iter, val_iter, test_iter = data.BucketIterator.splits((x_train, x_val, y_test), batch_size=batch_size, shuffle=True, repeat=False)



# Step 2 : RNN �� ����

# 2-1. model ���� �� �н�(train), ��(evaluate) �Լ� ����

class JW_GRU(nn.Module):

    # model�� init�� �Ű�� ������ ��Ÿ���� ��
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_classes, n_layers, dropout_ratio = 0.2 ):

        super(JW_GRU, self).__init__()

        # ���� �Է��� �ܾ� ������ ũ�� ������ ���� ���� -> �Ӻ��� ���̾� ��� �� �Ӻ��� ���� ������� ��ȯ
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim

        self.dropout = nn.Dropout(dropout_ratio)

        self.n_layers= n_layers

        # RNN �迭 ���� �Է��� embedding vector�� ����
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True )

        # ��º�
        self.out = nn.Linear(self.hidden_dim, n_classes)

    # rnn �迭 �н����� �ʱⰪ h_0�� �ʿ�
    def _init_state(self, batch_size=1):

        # iter : �ݺ� ������ ��ü�� ���� 
        # next : �ݺ� ���� ��ü�� ���� ��� ��ȯ 
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
        # many to one ���������� GRU ������ �ʱⰪ�� �������ָ� �� 
        # many to many�� ��� �ʱⰪ, �ʱⰪ�� �ƴ� ��츦 ������
        # GRU(x, h_0), GRU(x,h_t)�� ���� ���·� ���� �ؾ� �� 
        x, _= self.gru(x, h_0)

        #h_t_shape = [64, 256]
        h_t = x[:,-1,:]
        
        self.dropout(h_t)

        #output_shape = [64,2]
        logit = self.out(h_t)
        
        return logit


# �� ���� �� CUDA�� �н�
GRU_model = JW_GRU(vocab_size,128, 256, n_classes, 1, 0.5).to(device)
optimizer = optim.Adam(GRU_model.parameters(), lr= learning_rate)


# set train fucntion
def train(GRU_model, optimizer, train_iter):
    
    # train ����
    GRU_model.train()

    for _, batch in enumerate(train_iter):

        # �� ������ ���� �Ʒ� �����Ϳ� text, label field�� ���� �ϴ� �κ� 
        x, y = batch.text.to(device), batch.label.to(device)
    
        # y.data.sub_(1) : y.data ���� 1�� �A�� 
        # ���� y.data tensor�� 1,2�� �����Ǿ� ���� 
        # ���̺� ���� 0, 1 �� ��ȯ
        y.data.sub_(1)

        # �� batch ���� optimizer �ʱ�ȭ �ʿ� ( �� batch ���� �ٸ� ������ ������ input data�� Ȱ�� )
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

        # ����, �������� ���� cross entropy ������ ������ return ( default : mean )
        # �ش� ���������� mean, sum ����� ���� ���̴� �̺�
        loss = F.cross_entropy(logit, y, reduction='sum')

        # ��� loss�� �����ؾ� ��� loss ��� ���� 
        total_loss += loss.item()

        correct += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)

    avg_loss = total_loss / size

    avg_accuracy = 100.0 * correct / size

    return avg_loss, avg_accuracy


# 2-2. �н� with training data
best_val_loss = None

for _epoch in range(1, epoch+1):

    train(GRU_model, optimizer, train_iter)

    # 2-3. �� with validation data
    val_loss, val_accuracy = evaluate(GRU_model, val_iter)

    print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (_epoch, val_loss, val_accuracy))

    # min(val_loss) �ϋ��� �� ����
    if not best_val_loss or val_loss < best_val_loss:

        if not os.path.isdir("snapshot"):

            os.makedirs("snapshot")

        torch.save(GRU_model.state_dict(), './snapshot/txtclassification.pt')

        best_val_loss = val_loss


# testing

# load best performance models
GRU_model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))

# 2-4. �׽�Ʈ with test_data
test_loss, test_acc = evaluate(GRU_model, test_iter)

# show result
print('�׽�Ʈ ����: %5.2f | �׽�Ʈ ��Ȯ��: %5.2f' % (test_loss, test_acc))