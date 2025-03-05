import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
import re

# Setting a seed for reproducibility
torch.manual_seed(1)

# Creating the corpus
corpus = np.array([
    "Movies are fun for everyone",
    "Watching movies is great fun",
    "Enjoy a great movie today",
    "Research is interesting and important",
    "Learning math is very important",
    "Science discovery is interesting",
    "Rock is great to listen to",
    "Listen to music for fun",
    "Music is fun for everyone",
    "Listen to folk music"
])

# Defining a function that extracts the tokens from a string
def tokenize(text):
    return [token.lower() for token in re.findall("\w+", text)]

# Defining a function for extracting the vocabulary from a corpus
def get_vocabulary(corpus):
    vocabulary = [token for text in corpus for token in tokenize(text)]
    vocabulary = list(set(vocabulary))
    vocabulary = sorted(vocabulary)
    return vocabulary

# Defining a function that creates a document-term matrix
def get_DTM(corpus):
    lst = []
    for doc in corpus:
        
        lsti = []
        for token in get_vocabulary(corpus):
            if token in tokenize(doc):
                lsti.append(1.0)
            else:
                lsti.append(0.0)
        
        lst.append(lsti)
    return torch.tensor(lst, dtype=torch.float32)
    

# Creating the document-term matrix
dtm = get_DTM(corpus)
print(dtm)

# Labeling the documents (0: cinema, 1: science, 2: music)
df = pd.DataFrame({
    "doc": corpus,
    "class": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
})

# Creating the inputs and targets
x = dtm
y = torch.tensor(df["class"].to_numpy(), dtype=torch.long)


# Creating the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear1 = nn.Linear(x.size()[1], 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 3)
        
    def forward(self, x):
        x = self.linear1(x)
        x = f.relu(x)
        x = f.dropout(x, 0.2)
        
        x = self.linear2(x)
        x = f.relu(x)
        x = f.dropout(x, 0.2)
        
        x = self.linear3(x)
        return x
    
    
# Instantiating the model, optimizer and loss criterion
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss() 

# Training the model
for i in range(100):
    
    ypred = model(x)
    loss = criterion(ypred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"epoch: {i}, loss: {loss.item()}")

# Evaluating
with torch.no_grad():
    ypred = model(x)
    print(y)
    print(ypred.argmax(dim=1))
