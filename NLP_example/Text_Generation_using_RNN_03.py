# -*- coding: cp949 -*-

'''

 Author / date : Jinwoo Lee / 2022-08-28


< Word RNN ( N:M RNN ������ �̿��� ) >

 - pytorch �� embedding layer�� ����Ͽ� �ܾ� ������ Text Generation �����غ���


<Step>

1. Text Preprocessing

   1-1. ������ �ܾ�� tokenization

   1-2. �ܾ ���� ���� �Ӻ��� �� �ܾ����� ���� 

   1-3. sentence =>  word embedding vector�� ��ȯ ��

        �Է� & ���� ������ ���� 

        �Է� ������ : sentence[:-1] , ���� ������ : sentence[1:]




2. Construct Model

   2-1. embedding layer�� �߰� �� RNN �Ű�� ���� 

        raw_data -> embedding layer -> RNN_input -> hidden layer -> output layer


        * data�� ���� ��ȭ 

         if data.shape = [1,6] ( batch, time_step ) , after embedding layer then data.shape = [1,6,5] 
       
        why ? 
        
         =? 0, ... 5 �� �ش��ϴ� �� time step t�� ���Ͽ�  t = [ 1,2, 3.3 , 4.2 ...] �� ���� ũ�� 5(����� ����) ��ŭ�� embedding vector�� ��ȯ �ϱ� ���� 


        * RNN�� �Է� 
        
           x, _status= self.rnn(x) �� ���¸� ���� h_t�� ������� �ʴ´�

           => �̷������δ� ����ؾ� ������, default ������ ���� ������

              ������ task �ڵ������� ��������� h_t�� ����ϴµ�, text ����, �з� ��Ʈ������ �׷��� ���� �ʴ� ������ 

              ->  

              ->

   2-2. construct model and set loss, optimizer

   2-3. Training


'''


import torch
import torch.nn as nn
import torch.optim as optim


sentence = "Repeat is the best medicine for memory"


# Step 1

# 1-1 : word tokenization
words = list(set(sentence.split()))

# 1-2 : ���� �Ӻ��� �� �ܾ� ���� ����
vocab = { word : idx+1 for idx, word in enumerate(words) }
num2word = {idx+1 : word for idx, word in enumerate(words) }

# unknown token �� ���� ����
vocab['<unk>'] = 0
num2word[0] = '<unk>'


# 1-3 : sentence�� word embedding vector�� ��ȯ ( word �� tokenization �ϰ�, �ش� �������� ������ ���� �Ӻ��� ȭ) �� �Է�, ���� ������ ����

def make_data(sentence, vocab):

    sen_encoded = [vocab[word] for word in sentence.split()]

    _input, _label = sen_encoded[:-1] , sen_encoded[1:]

    # boath x_train.shape and label = [1,6] 
    # batch size�� �������� ���� �ٱ� ������ 1���� �߰�
    # NN�� input�� batch_size�� ����� 3D tensor�� Ȱ��
    x_train = torch.LongTensor(_input).unsqueeze(0)
    label = torch.LongTensor(_label).unsqueeze(0)
    
    return x_train, label

# train_data.shape , label.shape = [1,6]
train_data, label_data = make_data(sentence, vocab)


# Step 2

# model hyperparams
init_dim, output_dim = len(vocab), len(vocab)

# RNN�� input dim�� embedding vector�� ũ�⿡ ���� ���� �� 
rnn_input_dim, hidden_dim = 5, 15

learning_rate = 0.1


# 2-1 : embedding layer�� ���Ե� RNN ����

class JW_RNN(nn.Module):

    def __init__(self, init_dim, rnn_input_dim, hidden_dim, output_dim, batch_first = True ):

        super(JW_RNN, self).__init__()

        # raw data�� embedding layer�� ����ϸ�, ������ ������ ���� (rnn_input_dim) embedding vector�� ��ȯ
        self.embedding_layer = nn.Embedding(num_embeddings = init_dim, embedding_dim = rnn_input_dim)

        self.rnn = nn.RNN(rnn_input_dim, hidden_dim, batch_first = True )

        self.linear = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):

        # [1,6] -> [1,6,5]
        x = self.embedding_layer(x)
        
        # [1,6,5] -> [1,6,15]
        x, _status= self.rnn(x)
        
        # [1,6,15] -> [1,6,8]
        x  = self.linear(x)
        
        # 8���� ���� �ܾ� �ĺ���� ���ؾ� ������ 
        # ���ؾ� �ϴ� �ĺ� ������ ������ �������� concatenate �ϱ� ���� -1 
        x = x.view(-1, len(vocab))

        return x 


# 2-2 : construct model and set loss & optimizer
net = JW_RNN(init_dim, rnn_input_dim, hidden_dim, output_dim)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(params = net.parameters())


# 2-3 : Training

for epoch in range(201):

    optimizer.zero_grad()

    output = net(train_data)

    loss = loss_function(output, label_data.view(-1))

    loss.backward()

    optimizer.step()


    prediction_string = ""

    # records

    if epoch % 40 == 0 :

        print("[{:02d}/201] {:.4f} ".format(epoch+1, loss))

        pred = output.softmax(-1).argmax(-1).tolist()

 
        prediction_string = "Repeat"

        for value in pred:

            prediction_string += " " + num2word[value]

        print(prediction_string)
        print()
