import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchtext.vocab import GloVe


def read_data(file_name):
    with Path(f"./datas/twitter/{file_name}.txt").open() as f:
        lines = f.readlines()
    datas = [[[]], [[]]]  # [ sequences, tags(or origin sequence) ]
    submit = file_name == "test-submit"  # if submitting, change return tag to return origin word
    for line in lines:
        if line == "\n":  # new sequence
            datas[0].append([])
            datas[1].append([])
            continue
        sp = line.split()  # word, (tag)
        datas[0][-1].append(encode_word(sp[0]))
        datas[1][-1].append(sp[0] if submit else sp[1])
    return datas[0][:-1], datas[1][:-1]  # last one is empty
    # train, val: [["i", "am", "alvin"]...]    , [["o", "o", "b-person"]...]
    # submit:     [["i", "am", "<hashtag>"]...], [["i", "am", "#handsome"]...]


def encode_word(word):
    if word.startswith("@"):
        return "<at>"  # start with @
    if word.startswith("#"):
        return "<hashtag>"  # start with #
    if word.startswith("http") or word.startswith("www."):
        return "<url>"  # start with http or www.
    return re.sub(r"\W", "", word) or "<naw>"  # remove non-word, if empty then <naw>


class NERDataSet(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def create_pretrained_embedding(name, embedding_size):
    glove = GloVe(name=name, dim=embedding_size)
    mean_vector = glove.vectors.mean(dim=0).unsqueeze(dim=0)  # mean vector as default vector for new token

    def add_token(token):
        if glove.stoi.get(token, None) is not None:
            print(f'"{token}" is already in Glove')
            return
        glove.itos.append(token)
        glove.stoi[token] = len(glove.itos) - 1
        glove.vectors = torch.cat((glove.vectors, mean_vector), dim=0)
    add_token("<unk>")
    add_token("<pad>")
    add_token("<naw>")
    add_token("<hashtag>")
    add_token("<at>")
    add_token("<url>")
    return torch.nn.Embedding.from_pretrained(glove.vectors, freeze=True), glove.stoi
