from .BasicModule import BasicModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Bertsum_biGRU(BasicModule):
    def __init__(self, args):
        super(Bertsum_biGRU, self).__init__(args)
        self.model_name = 'Bertsum_biGRU'
        self.args = args
        
        I = args.embed_dim
        H = args.hidden_size   #số chiều của hidden_state == 768
        S = args.seg_num      #số đoạn văn.
        P_V = args.pos_num    #số trạng thái
        P_D = args.pos_dim    #số chiều của trạng thái
        self.abs_pos_embed = nn.Embedding(P_V,P_D)   #khởi tạo ngẫu nhiên 1 bộ embedding vị trí tương đối
        self.rel_pos_embed = nn.Embedding(S,P_D)    #khởi tạo ngẫu nhiên 1 bộ embedding vị trí tuyệt đối

        self.sent_RNN = nn.GRU(
                        input_size = I, #H == 768
                        hidden_size = H,   # H == 768
                        batch_first = True,
                        bidirectional = True
                        )
        self.fc = nn.Linear(2*H,2*H)   #đầu vào là H feature xong ra H feature

        # Parameters of Classification Layer
        self.content = nn.Linear(2*H,1,bias=False)  # đầu vào là H đầu ra là 1 .
        self.salience = nn.Bilinear(2*H,2*H,1,bias=False)   #đầu vào là 2 bộ feature đầu ra là 1 chiều
        self.novelty = nn.Bilinear(2*H,2*H,1,bias=False)   #tương tự như trên
        self.abs_pos = nn.Linear(P_D,1,bias=False)  #vào là P_D => đầu ra là 1
        self.rel_pos = nn.Linear(P_D,1,bias=False)  #đầu ra là 1 chiều
        self.bias = nn.Parameter(torch.FloatTensor(1).uniform_(-0.1,0.1)) # 1 số thực trong khoảng -0.1 đến 0.1

    def max_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]  #N = number of sentence, L = length of sentence , O_in = word_dim
        out = []
        for index,t in enumerate(x):  #index là câu thứ index trong a document
            
            #t = [L,O_in]
            t = t[:seq_lens[index],:]  #=> seq_lens là độ dài của 1 câu. Lấy từ từ đầu tiên cho đến hết độ dài câu
            
            #chuyển vị trước rồi thêm chiều t = [1,O_in,L]
            t = torch.t(t).unsqueeze(0)  #chuyển vị rồi thêm 1 chiều vào chiều số 0 => chiều 0 cũ thành chiều 1 mới, chiều 1 cũ thành chiều 2 mới
            
            #F.max_pool1d(t,t.size(2)) trả về [1,O_in,1]
            #sau mỗi vòng lặp thì thêm vào 1 tensor 3 chiều như trên đây là 1 list các tensor
            out.append(F.max_pool1d(t,t.size(2)))  #pool input t bằng cách lấy max của a phần tử với a là tham số truyền vào ở đây a  = t.size(2)
            # một cách không lặp lại VD: a = 5 thì xét từ 0 - 5 từ 6 - 11 ,...
            #tương tự 2D là xét ma trận
        
        #các phần tử con 3 chiều được kết nối theo hàng
        #mặc định ko có concat theo chiều nào thì sẽ cat theo chiều số 2. Ở đây là nối theo hàng
        #cắt chiều ngoài cùng và cat lại theo hàng thành 1 tensor 3 chiều
        #concat xong mới cắt chiều số 2
        out = torch.cat(out).squeeze(2) #bớt đi chiều số 2 nếu chiều thứ 2 có dim=1 còn ko thì không làm gì cả => chiều số 3 cũ thành chiều số 2....
        return out #=> [N,O_in]
    def avg_pool1d(self,x,seq_lens):
        # x:[N,L,O_in]
        out = []
        for index,t in enumerate(x):
            t = t[:seq_lens[index],:]  #[L,O_in]
            t = torch.t(t).unsqueeze(0)   #[1,O,L]
            out.append(F.avg_pool1d(t,t.size(2)))  #list các tensor 3 chiều : [1,O,1]
        
        out = torch.cat(out).squeeze(2)  #cắt chiều ngoài cùng và cat lại theo hàng thành 1 tensor 3 chiều
        return out   #=> [N,O_in]
    def forward(self,doc_lens,sent_embedding = None): #x truyền vào là 2 chiều. Hàng là các câu còn cột là các từ (N,L)
        sent_lens = 512                                       
        # word level GRU
        H = self.args.hidden_size  #768
        if sent_embedding is not None:
            word_out = sent_embedding
            
        # make sent features(pad with zeros)
        x = self.pad_doc(word_out,doc_lens)  
                                         
                                                                                #N = B = Batch = số doc train 1 lần
        # sent level GRU
        sent_out = self.sent_RNN(x)[0]                                           # (B,max_doc_len,H)
        #docs = self.avg_pool1d(sent_out,doc_lens)                               # (B,H)
        docs = self.max_pool1d(sent_out,doc_lens)                                # (B,H)
        probs = []
        for index,doc_len in enumerate(doc_lens):
            #lấy ma trận ở hàng thứ index
            valid_hidden = sent_out[index,:doc_len,:]       # (doc_len,H) #vứt đi các phần padding
            
            #đưa qua hàm tanh để đồng bộ giá trị biểu diễn các câu trong doc về -1;1 mà ko làm mất ngữ nghĩa
            doc = torch.tanh(self.fc(docs[index])).unsqueeze(0)         #   (1,H)
            s = Variable(torch.zeros(1,2*H))     #khởi tạo vector của summeries = vector 0. (1,H)
            if self.args.device is not None:
                s = s.cuda()    #vứt vào GPU để tính
            for position, h in enumerate(valid_hidden):         #position = vị trí của câu
                h = h.view(1, -1)                                       # (1,H)
                # get position embeddings
                abs_index = Variable(torch.LongTensor([[position]])) #variable hóa
                if self.args.device is not None:
                    abs_index = abs_index.cuda()      #vứt vào GPU tính
                abs_features = self.abs_pos_embed(abs_index).squeeze(0)   #vector  [P_D] position_dim
                #thử vị trí tương đối bằng hàm cos sin
                rel_index = int(round((position + 1) * 9.0 / doc_len)) #cái vị trí tuyệt đối tính sao ta
                rel_index = Variable(torch.LongTensor([[rel_index]]))
                if self.args.device is not None:
                    rel_index = rel_index.cuda()
                rel_features = self.rel_pos_embed(rel_index).squeeze(0)   #vector [P_D]
                
                # classification layer
                content = self.content(h)  #ra 1 số
                salience = self.salience(h,doc)  # ra 1 số
                novelty = -1 * self.novelty(h,torch.tanh(s))   #1 số
                abs_p = self.abs_pos(abs_features)   #1 số
                rel_p = self.rel_pos(rel_features)    # 1 số
                prob = torch.sigmoid(content + salience + novelty + abs_p + rel_p + self.bias) #xác suất 1 câu
                #[1,1] nhân ma trận với [1,2*H] => s = [1,2*H]
                s = s + torch.mm(prob,h)   #nhân 2 ma trận prob với h theo quy tắc nhân matrix của đại số tuyến tính
                probs.append(prob)    #list xác suất các câu của 1 document
        return torch.cat(probs).squeeze() #matrix [1,doc_len]  ma trận xác suất câu của cả mini_batch