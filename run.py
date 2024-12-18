import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

seed = 27
torch.manual_seed(seed)

# Load data
ptb = load_dataset("ptb-text-only/ptb_text_only") 
train = ptb["train"]["sentence"]
val = ptb["validation"]["sentence"]
test = ptb["test"]["sentence"]

# Takes in sentence and converts it to numbers
class numDataset(Dataset):
    def __init__(self, x, token_vocab=None, training=True):

        if training:
            self.token_vocab = {"<pad>":0, "<unk>": 1, "<s>": 2, "</s>": 3}

            for sent in x:
                for token in sent.split():
                    if token not in self.token_vocab:
                        # Encode position when adding token into vocab (using length)
                        self.token_vocab[token] = len(self.token_vocab)
        else:
            assert token_vocab is not None
            self.token_vocab = token_vocab

        self.corpus_token_ids = []
        self.targets = []
        for sent in x:
            # Add start and stop symbol to sentences
            token_ids = [self.token_vocab["<s>"]] + [self.token_vocab.get(token, self.token_vocab["<unk>"]) for token in sent.split()] + [self.token_vocab["</s>"]]
            self.corpus_token_ids.append(torch.tensor(token_ids))

            # Add padding instead of start symbol and to the end to make sure the target reflects the next token in the sequence
            targ_ids = [self.token_vocab.get(token, self.token_vocab["<unk>"]) for token in sent.split()] + [self.token_vocab["</s>"]] + [self.token_vocab["<pad>"]]
            self.targets.append(torch.tensor(targ_ids))


    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.targets[idx]


# Modify DataLoader's collate_fn to handle padded sequences
def padding(batch):

    token_ids = [item[0] for item in batch]
    targ_ids = [item[1] for item in batch]
    
    # Pad sequences using pre-defined pad token in vocab
    padded_sentences = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab["<pad>"])
    padded_targs = pad_sequence(targ_ids, batch_first=True, padding_value=train_dataset.token_vocab["<pad>"])

    return padded_sentences, padded_targs 

# Learn positional embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=5000):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, n_embd)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).view(1, x.size(1))
        embedding = self.pos_embedding(pos)
        return x + embedding

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = PositionalEncoding(n_embd)

        # A stack of encoder layers: transformer
        encoder_layer = nn.TransformerEncoderLayer(n_embd, n_head, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layer)

        self.fc_out = nn.Linear(n_embd, vocab_size)

    def forward(self, x): # x -> (batch_size, seq_len)

        # Padding mask
        pad_mask = (x == train_dataset.token_vocab["<pad>"])

        x = self.embedding(x)  # x -> (batch_size, seq_len, num_embd)
        x = self.pos_embedding(x)  # x -> (batch_size, seq_len, num_embd)

        # Create mask to prevent model from looking ahead
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        out = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask)
        
        # Project the output 
        out = self.fc_out(out) 

        return out


train_dataset = numDataset(train, training=True)
val_dataset = numDataset(val, token_vocab=train_dataset.token_vocab, training=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=padding)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=padding)


# Train model
n_embd = 128
vocab_size = len(train_dataset.token_vocab)  # 10001 (including </s> token)
n_head = 8
n_layer = 4
model = TransformerLM(vocab_size=vocab_size,
                      n_embd=n_embd,
                      n_head=n_head,
                      n_layer=n_layer)
pad_index = train_dataset.token_vocab["<pad>"]
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
model_path = os.path.join(os.path.dirname(__file__), "trained_model")
token_lookup = {idx: token for token, idx in train_dataset.token_vocab.items()}
num_epoch = 5
prev_loss = None
for epoch in range(num_epoch):

    epoch_loss = 0
    model.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        x_batch = x_batch
        y_batch = y_batch
        preds = model(x_batch)
        loss = loss_fn(preds.view(-1, preds.shape[-1]), y_batch.view(-1).long())
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
    print(f"Epoch Training Loss: {epoch_loss/len(train_loader)}")

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        perplexities = []
        for x_val, y_val in val_loader:
            x_val = x_val
            y_val = y_val

            # Calculate validation perplexity
            preds = model(x_val)
            for sent_preds, targ_preds in zip(preds, y_val):    
                loss = loss_fn(sent_preds, targ_preds.long())
                perplexities.append(torch.exp(loss).item())

            outputs = preds.view(-1, preds.shape[-1])
            targs = y_val.view(-1).long()
            loss = loss_fn(outputs, targs)
            total_val_loss += loss.item()
            
        print(f"Avg Validation Perplexity: {sum(perplexities)/ len(perplexities)}")
        val_loss = total_val_loss/ len(val_loader)
        print(f"Epoch Validation Loss: {val_loss}")
        # Save model with lowest loss
        if prev_loss is None or val_loss < prev_loss:
            print("LOWEST LOSS : SAVING MODEL")
            torch.save(model.state_dict(), model_path)
            prev_loss = val_loss

# Run on test data
output = os.path.join(os.path.dirname(__file__), "submission.csv")
saved_model = TransformerLM(vocab_size=vocab_size,
                      n_embd=n_embd,
                      n_head=n_head,
                      n_layer=n_layer)
saved_model.load_state_dict(torch.load(model_path))
saved_model = saved_model
saved_model.eval()
test_dataset = numDataset(test, token_vocab=train_dataset.token_vocab, training=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=padding)
sent_perplexity = []
for x_batch, y_batch in test_loader: 
    x_batch = x_batch
    y_batch = y_batch
    preds = saved_model(x_batch)
    
    for sent_preds, targ_preds in zip(preds, y_batch):
        loss = loss_fn(sent_preds, targ_preds.long())
        sent_perplexity.append(torch.exp(loss).item())
print(f"Avg Test Perplexity: {sum(sent_perplexity)/ len(sent_perplexity)}")

df = pd.DataFrame()
df["ppl"] = sent_perplexity
df.index = df.index
df.index.name = "ID"
df.to_csv(output)

