import torch


class process():
    def __init__(self):
        self.CLS_TOKEN = '[CLS]'
        self.SEP_TOKEN = '[SEP]'
        
        
    
    def make_features(self,batch,tokenizer,model_bert,doc_trunc=100,split_token='\n'):
        sents_list,targets,doc_lens = [],[],[]
        for doc,label in zip(batch['doc'],batch['labels']):
            sents = doc.split(split_token) #list sentences
            labels = label.split(split_token) 
            labels = [int(l) for l in labels] # list label
            max_sent_num = min(doc_trunc,len(sents)) #update max number sentences
            sents = sents[:max_sent_num]   #không để đoạn văn vượt quá 100
            labels = labels[:max_sent_num] #không để đoạn văn dài quá 100 câu
            sents_list += sents            #list sentences của 1 batch
            targets += labels               #list label của 1 batch
            doc_lens.append(len(sents))     #list len of document
        
        # trunc or pad sent
        batch_sents = []
        for sent in sents_list:
          
          # Add the special tokens.
          marked_text = "[CLS] " + sent + " [SEP]"
          batch_sents.append(marked_text)
        
        batch_sent_embeds = []
        for i,sent in enumerate(batch_sents):
            # Split the sentence into tokens.
            
            tokenized_text = tokenizer.tokenize(sent)
            if len(tokenized_text) > 511:
              tokenized_text = tokenized_text[:511]
              tokenized_text.append('[SEP]')
            # Map the token strings to their vocabulary indeces.
            # print(tokenized_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            # print(tokenized_text)
            temp = i%2
            segment_ids = [temp] * len(tokenized_text)

            tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
            segments_tensor = torch.tensor([segment_ids]).to('cuda')

            model = model_bert.to('cuda')
            
            with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensor)
            # `encoded_layers` has shape [12 x 1 x 22 x 768]

            # `token_vecs` is a tensor with shape [22 x 768]
            token_vecs = encoded_layers[11][0]
            # Calculate the average of all 22 token vectors.
            sentence_embedding = torch.mean(token_vecs, dim=0) # torch.Size([768])
            #tăng chiều
            sentence_embedding = sentence_embedding.unsqueeze(0)
            batch_sent_embeds.append(sentence_embedding)
        batch_sent_embeds = torch.cat((batch_sent_embeds),dim=0)
        # print(batch_sent_embeds.shape)
        # print('number all sent of batch = ',len(targets))
        # print('number of sent in first doc = ',doc_lens)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']

        return batch_sent_embeds,targets,summaries,doc_lens