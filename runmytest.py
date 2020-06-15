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
# import pytorch_pretrained_bert.modeling as modeling

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
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
parser.add_argument('-test_dir',type=str,default='../data/test_cnn_dailymail.json')
parser.add_argument('-ref',type=str,default='outputs/ref/') #references
parser.add_argument('-hyp',type=str,default='outputs/hyp/') #hypothe
parser.add_argument('-filename',type=str,default='x.txt') # TextFile to be summarized
parser.add_argument('-topk',type=int,default=3) #top các câu xác suất cao nhất
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
# # config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=240,
# #                 num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
# # model_bert = modeling.BertModel(config=config) #nạp mô hình theo tham số của mình
# # model_bert = BertModel.from_pretrained('bert-base-uncased') #nạp mô hình theo tham số mặc định của mô 
model_bert = BertModel.from_pretrained('bert-base-uncased')

model_bert.eval()
# model_bert.cuda()



def test():
    print('running test!')
    with open(args.test_dir,encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    test_dataset = dataset.dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
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
    print(net)
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        sent_embed,labels,summaries,doc_lens = process.process.make_features(process,batch,tokenizer,model_bert)
        t1 = time()
        if use_gpu:
            probs = net( doc_lens,sent_embed.cuda())
        else:
            probs = net(doc_lens,sent_embed)
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            prob_n = prob.cpu().data.numpy()
            topk_indices = np.where(prob_n > 0.6)
            # print(topk_indices)
            if len(topk_indices[0]) > 4:
                topk_index = topk_indices[0][:4]
                topk_index = sorted(topk_index)
            else:
                topk_index = topk_indices[0]
                topk_index = sorted(topk_index)
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            hyp = [doc[index] for index in topk_index]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref,str(file_id)+'.txt'), 'w',encoding='utf-8') as f:
                f.write(ref)
            with open(os.path.join(args.hyp,str(file_id)+'.txt'), 'w',encoding='utf-8') as f:
                f.write('.\n'.join(hyp))
            start = stop
            file_id = file_id + 1
        t2 = time()
        time_cost += t2 - t1
    print('Speed: %.2f docs / s' % (doc_num / time_cost))


if __name__=='__main__':
    test()