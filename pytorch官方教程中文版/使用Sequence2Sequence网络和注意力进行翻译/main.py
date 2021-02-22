from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from colorama import Style, Fore, Back
from icecream import ic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 小写，修剪和删除非字母字符
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# 2.1 读取数据文件
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # 读取文件并分成几行
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

    # 将每一行拆分成对并进行标准化
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # 反向对，使Lang实例
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH and \
           p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# 准备数据的完整过程是：
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

print(random.choice(pairs))


# 3.Seq2Seq模型

# 3.1 编码器   input: fra   output: eng
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)   # 输入hidden_size(来自embedding), 输出hidden_size

    def forward(self, input, hidden):
        # print('\nEncoderRNN--input', input.shape)  # torch.Size([1])
        # print('EncoderRNN--hidden', hidden.shape)  # torch.Size([1, 1, 256])
        embedded = self.embedding(input).view(1, 1, -1)  # torch.Size([1, 1, 256])
        # print('EncoderRNN--embedded', embedded.shape)

        output = embedded
        output, hidden = self.gru(output, hidden)
        # print('EncoderRNN--output', output.shape)  # torch.Size([1, 1, 256])
        # print('EncoderRNN--hidden', hidden.shape)  # torch.Size([1, 1, 256])
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 3.2 解码器   input: fra   output: eng

# class DecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.gru = nn.GRU(hidden_size, hidden_size)
#
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, input, hidden):
#         output = self.embedding(input).view(1, 1, -1)
#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)


# 3.3 注意力机制解码器   input: fra   output: eng   MAX_LENGTH: 10
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)   # 输入来自embedding  输出：hidden_size
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param input: 如果不是Force Teaching ，input就是每一步的预测结果
        :param hidden: 就是从编码器传来的hidden shape(1,1,256)
        :param encoder_outputs: 就是编码器的输出  shape(10,256)
        :return:
        """
        # print('\nAttnDecoderRNN--input', input.shape)  # torch.Size([1, 1])
        # print('AttnDecoderRNN--hidden', hidden.shape)  # torch.Size([1, 1, 256])
        # print('AttnDecoderRNN--encoder_outputs', encoder_outputs.shape)  # torch.Size([10, 256])
        embedded = self.embedding(input).view(1, 1, -1)  # torch.Size([1, 1, 256])
        # print('AttnDecoderRNN--embedded', embedded.shape)

        embedded = self.dropout(embedded)
        # print('AttnDecoderRNN--torch.cat((embedded[0], hidden[0]), 1))', torch.cat((embedded[0], hidden[0]), 1).shape)   # torch.Size([1, 512])
        # print('AttnDecoderRNN--self.attn(torch.cat((embedded[0], hidden[0]), 1))', self.attn(torch.cat((embedded[0], hidden[0]), 1)).shape)   # torch.Size([1, 10])

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)   # torch.Size([1, 10])  当前输入和以往hidden产生注意力权重，hidden中包含重要的特征，当前输入占重要特征

        # attn_weights.unsqueeze(0).shape: torch.Size([1, 1, 10])
        # encoder_outputs.unsqueeze(0).shape: torch.Size([1, 10, 256])
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))   # torch.Size([1, 1, 256])

        output = torch.cat((embedded[0], attn_applied[0]), 1)   # torch.Size([1, 512])  本来是embedded，现在有了attn_applied，attn_applied可以让网络在这一步可以更好的理解这个单词对应于input中的哪个单词

        output = self.attn_combine(output).unsqueeze(0)  # torch.Size([1, 1, 256])

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # print('AttnDecoderRNN--output', output.shape)   # torch.Size([1, 1, 256])
        # print('AttnDecoderRNN--hidden', hidden.shape)   # torch.Size([1, 1, 256])

        output = F.log_softmax(self.out(output[0]), dim=1)   # torch.Size([1, 2803])

        return output, hidden, attn_weights


def initHidden(self):
    return torch.zeros(1, 1, self.hidden_size, device=device)


# 4.1 准备训练数据

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# 4.2 训练模型

teacher_forcing_ratio = 0.5

# input_tensor.shape (n, 1)  target_tensor.shape (n, 1)  MAX_LENGTH：10(一个句子中最大的单词个数)
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()   # torch.Size([1, 1, 256]) 这里为什么是1，1，256？ 有可能是batch_size:1 每次1个单词

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)  # 输入句子中单词的个数
    target_length = target_tensor.size(0)  # 输出句子中单词的个数

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)  # 10，256

    loss = 0

    # 训练一个完整的输入句子  ++++编码阶段++++
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)   # input_tensor[ei]：一个单词对应的索引; encoder_output.shape: torch.Size([1, 1, 256])；encoder_hidden.shape: torch.Size([1, 1, 256])
        encoder_outputs[ei] = encoder_output[0, 0]  # encoder_outputs.shape: torch.Size([10, 256])  10指的是一个输入句子的单词数量

    # 初始化（解码器）输入
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # encoder_hidden中保存着输入句子重要的信息，然后传递给解码器
    decoder_hidden = encoder_hidden

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False

    # 训练一个完整的输出序列  ++++解码阶段++++  >>>>>>注意力解码器
    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:  # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            topv, topi = decoder_output.topk(1)  # topv是概率，topi是索引
            # ic(decoder_output, decoder_output.topk(1), topi.squeeze(), topi.squeeze().detach())
            decoder_input = topi.squeeze().detach()  # detach from history as input
            # ic(target_tensor[di])

            loss += criterion(decoder_output, target_tensor[di])

            # 如果下一步的输入（当前的预测）等于EOS_TOKEN,那么退出循环，感觉这一步可要可不要
            # if decoder_input.item() == EOS_token:  
                # break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# 辅助函数

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()

    plot_losses = []

    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]

    print(Fore.LIGHTBLUE_EX)
    print('trainIters>>>i')
    print(training_pairs[0])
    print(Style.RESET_ALL)


    criterion = nn.NLLLoss()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


    showPlot(plot_losses)


# 结果绘图函数

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# 评价函数

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# 5.训练和评价

hidden_size = 256

# 实例化编码器
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)

# 实例化注意力解码器
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

print('\ninput_lang.n_words', input_lang.n_words)  # 4345  fra
print('\noutput_lang.n_words', output_lang.n_words)  # 2803  eng

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)