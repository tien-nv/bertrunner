import torch
from torch.autograd import Variable
class BasicModule(torch.nn.Module):

    def __init__(self, args):
        super(BasicModule,self).__init__()
        self.args = args
        self.model_name = str(type(self))

#word_out = [B,H] với B là số doc, H là biểu diễn của 1 câu với H =768 hoặc 3072 tùy theo hướng sử dụng
#word_out là 1 tensor 2 chiều tương ứng với 1 hàng là 1 câu và câu đó có 768 đặc trưng hoặc là 3072
    def pad_doc(self,words_out,doc_lens): 
        pad_dim = words_out.size(1) #768 or 3072 tùy theo cách sử dụng
        max_doc_len = max(doc_lens)   #đồng bộ theo độ dài của document dài nhất tính theo số câu
        sent_input = []
        start = 0
        for doc_len in doc_lens: #doc_lens: list chứa độ dài các câu
            stop = start + doc_len
            valid = words_out[start:stop]                      # (doc_len,768) #doc_len = số sentence/doc
            start = stop
            if doc_len == max_doc_len:
                sent_input.append(valid.unsqueeze(0))  #list gồm các tensor [1,max_len,2*H]
            else:
                # pad = [max_doc-doc_len,2*H]
                pad = Variable(torch.zeros(max_doc_len-doc_len,pad_dim))    #biến thành biến để có thể đạo hàm...
                if self.args.device is not None:
                    pad = pad.cuda()
                sent_input.append(torch.cat([valid,pad]).unsqueeze(0))          # (1,max_len,2*H)
        sent_input = torch.cat(sent_input,dim=0)                                # (B,max_len,2*H)
        return sent_input
    def autosave(self):
        checkpoint = {'model':self.state_dict(), 'args': self.args}
        best_path = '%s%s_seed_%d.pt' % (self.args.auto_save_dir,self.model_name,self.args.seed) #GRU_GRU_seed_1.pt
        torch.save(checkpoint,best_path)

        return best_path
    def save(self):
        checkpoint = {'model':self.state_dict(), 'args': self.args}
        best_path = '%s%s_seed_%d.pt' % (self.args.save_dir,self.model_name,self.args.seed) #GRU_GRU_seed_1.pt
        torch.save(checkpoint,best_path)

        return best_path

    def load(self, best_path):
        if self.args.device is not None:
            data = torch.load(best_path)['model']
        else:
            data = torch.load(best_path, map_location=lambda storage, loc: storage)['model']
        self.load_state_dict(data)
        if self.args.device is not None:
            return self.cuda()
        else:
            return self