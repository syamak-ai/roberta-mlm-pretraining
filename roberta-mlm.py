#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import numpy as np

import torch
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    pipeline
)
from torch.utils.data import random_split, Dataset, DataLoader

MAX_LEN = 35
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = torch.Generator().manual_seed(42)

byte_tokenizer = ByteLevelBPETokenizer()
byte_tokenizer.train(files='kant.txt', vocab_size=52000, min_frequency=4, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>"
])
byte_tokenizer.save_model('./')

byte_tokenizer = ByteLevelBPETokenizer(
    "vocab.json",
    "merges.txt",
)

for i, (token, token_id) in enumerate(byte_tokenizer.get_vocab().items()):
    if i<5:
        print(token_id, repr(token), repr(byte_tokenizer.decode([token_id])))

byte_tokenizer._tokenizer.post_processor =  RobertaProcessing(
    ("</s>", byte_tokenizer.token_to_id("</s>")),
    ("<s>", byte_tokenizer.token_to_id("<s>")),
)

roberta_tokenizer = RobertaTokenizer.from_pretrained("./", max_length=MAX_LEN)

config = RobertaConfig(
    hidden_size=256,
    vocab_size=len(roberta_tokenizer),
    max_position_embeddings=MAX_LEN+2,
    num_attention_heads=8,
    num_hidden_layers=4,
    type_vocab_size=1,
    intermediate_size=1024
)
model = RobertaForMaskedLM(config=config)
model.to(DEVICE);

length = []

with open("./kant.txt", "r", encoding="utf-8") as f:
    for line in tqdm(f):
        tokens = roberta_tokenizer(
            line,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        length.append(len(tokens))

print(np.percentile(length, [50, 75, 90, 95, 99, 100]))

print("Tokenizer vocab size:", len(roberta_tokenizer))
print("Model vocab size:", model.config.vocab_size)

class KantDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(file_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.lines[idx],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

dataset = KantDataset(
    "./kant.txt",
    roberta_tokenizer,
    MAX_LEN
)

n = len(dataset)
n_val = max(1, int(0.1 * n))
n_train = n - n_val
train_dataset, val_dataset = random_split(
    dataset,
    [n_train, n_val],
    generator=generator
)

data_collator = DataCollatorForLanguageModeling(
tokenizer=roberta_tokenizer, mlm=True, mlm_probability=0.15
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5
)

def val_loss(loader):
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            output = model(**batch)
            total_loss += output.loss.item()

    valid_loss = total_loss / len(loader)
    return valid_loss

for epoch in range(3):
    model.train()
    total_loss=0
    for i, batch in enumerate(tqdm(train_loader)):
     
        input_ids=batch["input_ids"].to(DEVICE)
        att_mask=batch['attention_mask'].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        
        output = model(
            input_ids=input_ids,
            attention_mask=att_mask,
            labels=labels
        )

        loss = output.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    print(f"train_loss: {total_loss / len(train_loader):.4f}")
    print(f"valid_loss: {val_loss(val_loader):.4f}")    

model.save_pretrained("./kant_roberta")
roberta_tokenizer.save_pretrained("./kant_roberta")

fill_mask = pipeline(
    "fill-mask",
    model="./kant_roberta",
    tokenizer="./kant_roberta",
    device=DEVICE
)

result = fill_mask("Human thinking involves human<mask>.")

for prediction in result:
    print(prediction)