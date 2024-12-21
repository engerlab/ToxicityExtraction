from cryptography.fernet import Fernet
from transformers import AutoModel, AutoConfig
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import random
import utils
from utils import chunk_spans, evaluate


#custom_key=input('Key?:').encode()
custom_key='ecQy-kYU4ywr6VJMOzr_hKl4dNsZjYPaQRWTWFxONxY='.encode()
cipher = Fernet(custom_key)
with open('all_samples.p', 'rb') as file:
    encrypted_data = file.read()
    decrypted_data = pickle.loads(cipher.decrypt(encrypted_data))

all_samples=decrypted_data
del decrypted_data



class NER(torch.nn.Module):
    def __init__(self,language_model):
        super(NER, self).__init__()
        self.config = AutoConfig.from_pretrained(language_model)
        self.lm=AutoModel.from_pretrained(language_model)
        self.num_classes = 3
        self.projection=nn.Linear(self.config.hidden_size,self.num_classes)

    def forward(self,input_ids):
        hiddens=self.lm(input_ids)
        return self.projection(hiddens['last_hidden_state']).squeeze()




all_keys=list(all_samples.keys())


random.shuffle(all_keys)



train_ids=all_keys[0:342]
test_ids=all_keys[342:]

train_set=list()
test_set=list()

for key in train_ids:
    train_set=train_set+all_samples[key]
for key in test_ids:
    test_set=test_set+all_samples[key]

lm_version='Bio_ClinicalBERT'
#lm_version='Clinical-Longformer'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model=NER(lm_version).to(device)
optimizer=Adam(model.parameters(),lr=0.00001)
loss_function=nn.CrossEntropyLoss()


n_epoch=3

for epoch in range(n_epoch):
    all_loss=list()
    for sample in train_set:
        model.zero_grad()
        max_len=model.config.max_position_embeddings

        token_ids=torch.tensor([sample.token_ids[0:max_len]],dtype=torch.long).to(device)
        pred=model(token_ids)
        target=torch.tensor(sample.labels[0:max_len],dtype=torch.long).to(device)
        loss=loss_function(pred,target)
        all_loss.append(loss.detach().cpu().item())
        loss.backward()
        optimizer.step()

    print('Average loss=',np.mean(all_loss))


with torch.no_grad():
    model.eval()

    overlap_scores={'p':[],'r':[],'f1':[]}
    strict_scores = {'p': [], 'r': [], 'f1': []}
    

    for sample in test_set:
        max_len = model.config.max_position_embeddings
        token_ids = torch.tensor([sample.token_ids[0:max_len]], dtype=torch.long).to(device)
        pred=model(token_ids)

        pred=list(torch.argmax(pred,dim=1).detach().cpu().numpy())

        diff=max(len(sample.token_ids)-max_len,0)
        pred=pred+[0]*diff

        pred=[(sample.token_spans[span[0]][0],sample.token_spans[span[1]][1],'SideEffect') for span in chunk_spans(pred)]
        gold=[(sample.spans[key][1][0],sample.spans[key][1][1],sample.spans[key][0]) for key in [key for key in sample.spans]]

        if len(pred)==0 and len(gold)==0:
            continue

# Example usage
#gold_spans = [(0, 5), (10, 15), (20, 25)]  # Gold standard spans
#predicted_spans = [(0, 5), (10, 15), (20, 23)]  # Predicted spans


#results = evaluate_ner(gold_spans, predicted_spans)
#print("Strict Measures:", results["strict"])
#print("Overlapping Measures:", results["overlapping"])




