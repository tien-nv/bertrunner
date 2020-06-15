#cần phải có file __init__.py để thông báo đó là 1 package.
# from preprocess import word2vec as w2v
# w2v.word2vec('c')
from preprocess import dataset,process
import models

import json
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from time import time
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pytorch_pretrained_bert.modeling as modeling
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-auto_save_dir',type=str,default='checkpoint/')
parser.add_argument('-embed_dim',type=int,default=768)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=100)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='Bertsum_biGRU')
parser.add_argument('-hidden_size',type=int,default=200) #hoặc 3072 nếu concatnate 4 layer cuối
# train
parser.add_argument('-lr',type=float,default=1e-3) #learning rate
parser.add_argument('-batch_size',type=int,default=32)
parser.add_argument('-epochs',type=int,default=4)
parser.add_argument('-seed',type=int,default=1) #random
parser.add_argument('-train_dir',type=str,default='../data/train_cnn_dailymail.json')
parser.add_argument('-val_dir',type=str,default='../data/val_cnn_dailymail.json')
parser.add_argument('-report_every',type=int,default=1500) #báo cáo result sau một số bước
parser.add_argument('-seq_trunc',type=int,default=50) 
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/Bertsum_biGRU_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='../data/test.json')
parser.add_argument('-ref',type=str,default='outputs/ref/') #references
parser.add_argument('-hyp',type=str,default='outputs/hyp/') #hypothe
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=8) #top các câu xác suất cao nhất
# device
parser.add_argument('-device',type=int,default=0)
# option
parser.add_argument('-test',action='store_false')
parser.add_argument('-debug',action='store_true')
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 



logging.info('Loading bert.Wait a second,please')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=240,
#                 num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
# model_bert = modeling.BertModel(config=config) #nạp mô hình theo tham số của mình
model_bert = BertModel.from_pretrained('bert-base-uncased') #nạp mô hình theo tham số mặc định của mô 

model_bert.eval()

def eval(net,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    print('start validation data')
    for batch in data_iter:
        sents_embed,targets,_,doc_lens = process.process.make_features(process,batch,tokenizer,model_bert)  
        sents_embed,targets = Variable(sents_embed), Variable(targets.float())
        if use_gpu:
            sents_embed= sents_embed.cuda()
            targets = targets.cuda()
        probs = net(doc_lens,sents_embed)
        loss = criterion(probs,targets) #binary classifier loss
        total_loss += loss.item()
        batch_num += 1  #số lượng mini_batch trong 1 lần train
        if batch_num%100 == 0:
            print('Validation_batch: ',batch_num)
    loss = total_loss / batch_num  #tổng giá trị hàm loss chia cho trung bình
    net.train()
    return loss

def train():
    logging.info('Loading train and val dataset.Wait a second,please')
    
    # with open(args.word2id) as f:
    #     word2id = json.load(f)
    # vocab = preprocess.vocab.vocab(embed, word2id)


    with open(args.train_dir,encoding='utf-8') as f: #train.json
        examples = [json.loads(line) for line in f]
    train_dataset = dataset.dataset(examples)

    with open(args.val_dir,encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    val_dataset = dataset.dataset(examples)
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False)
    # loss function
    criterion = nn.BCELoss() #binary cross entropy
    # update args
    # args.embed_num = embed.size(0)
    # args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    net = getattr(models,args.model)(args)  #load Bertsum_biGRU
    if use_gpu:
        net.cuda()
    
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))
    print(net.parameters())

    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr)
    net.train()
    
    t1 = time() 
    for epoch in range(1,args.epochs):
        for i,batch in enumerate(train_iter):
            sent_embeds,targets,_,doc_lens = process.process.make_features(process,batch,tokenizer,model_bert)
            sent_embeds,targets = Variable(sent_embeds), Variable(targets.float())
            if use_gpu:
                sent_embeds = sent_embeds.cuda()
                targets = targets.cuda()
            probs = net(doc_lens,sent_embeds)
            loss = criterion(probs,targets)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm(net.parameters(), args.max_norm)
            optimizer.step()
            if i%200 == 0:
                print('Batch ID:%d Loss:%f' %(i,loss.data.item()))
                # continue
            if i % args.report_every == 0:
                print("caculate val")
                cur_loss = eval(net,val_iter,criterion)
                print("caculate val done")
                if cur_loss < min_loss:  #min_loss = vô cùng
                    min_loss = cur_loss
                    net.save()
                logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                        % (epoch,min_loss,cur_loss))
    t2 = time()
    logging.info('Total Cost:%f h'%((t2-t1)/3600))
    net.autosave()

# def test():
#     print('running test!')
#     with open(args.test_dir,encoding='utf-8') as f:
#         examples = [json.loads(line) for line in f]
#     test_dataset = dataset.dataset(examples)

#     test_iter = DataLoader(dataset=test_dataset,
#                             batch_size=args.batch_size,
#                             shuffle=False)
#     if use_gpu:
#         checkpoint = torch.load(args.load_dir)
#     else:
#         checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

#     # checkpoint['args']['device'] saves the device used as train time
#     # if at test time, we are using a CPU, we must override device to None
#     if not use_gpu:
#         checkpoint['args'].device = None
#     net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
#     net.load_state_dict(checkpoint['model'])
#     if use_gpu:
#         net.cuda()
#     net.eval()
    
#     doc_num = len(test_dataset)
#     time_cost = 0
#     file_id = 1
#     for batch in tqdm(test_iter):
#         sent_embed,labels,summaries,doc_lens = process.process.make_features(process,batch,tokenizer,model_bert)
#         t1 = time()
#         if use_gpu:
#             probs = net( doc_lens,Variable(sent_embed).cuda())
#         else:
#             probs = net(doc_lens,Variable(sent_embed))
#         y_pred = np.zeros(len(probs))
#         for i,j in enumerate(probs):
#             if (i >= 0,5):
#                 y_pred[i] = 1
#         y_true = labels
#         print('accuracy = ' ,accuracy_score(y_true,y_pred))
#         t2 = time()
#         time_cost += t2 - t1
#         # start = 0
#     #     for doc_id,doc_len in enumerate(doc_lens):
#     #         stop = start + doc_len
#     #         prob = probs[start:stop]
#     #         # print(prob)
            
#     #         # label = labels[start:stop]
#     #         # prob_n = prob.cpu().data.numpy()
            
           
            
            
#     #         topk = min(args.topk,doc_len)
#     #         topk_indices = prob.topk(4)[1].cpu().data.numpy()
#     #         topk_indices.sort()
#     #         doc = batch['doc'][doc_id].split('\n')[:doc_len]
#     #         hyp = [doc[index] for index in topk_indices]
#     #         ref = summaries[doc_id]
#     #         with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w',encoding='utf-8') as f:
#     #             f.write(ref)
#     #         with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w',encoding='utf-8') as f:
#     #             f.write('\n'.join(hyp))
#     #         start = stop
#     #         file_id = file_id + 1
#     # print('Speed: %.2f docs / s' % (doc_num / time_cost))


# def predict(examples):
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = preprocess.vocab(embed, word2id)
    pred_dataset = preprocess.dataset(examples)

    pred_iter = DataLoader(dataset=pred_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(pred_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(pred_iter):
        features, doc_lens = vocab.make_predict_features(batch)
        t1 = time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            topk = min(args.topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch[doc_id].split('. ')[:doc_len]
            hyp = [doc[index] for index in topk_indices]
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w') as f:
                f.write('. '.join(hyp))
            start = stop
            file_id = file_id + 1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))

if __name__=='__main__':
    train()