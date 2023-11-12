from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, unpad_sequence
from torch.utils.tensorboard import SummaryWriter

from conlleval import evaluate
from NERDataLoader import NERDataLoader
from NERDataSet import read_data


class Net(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, bidirectional=True, dropout=0.5):
        super().__init__()
        self.gru = torch.nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(2 * hidden_size if bidirectional else hidden_size, output_size, bias=False)

    def forward(self, batch_seq):
        seq, _ = self.gru(batch_seq)  # (packed_sum_seq_len, embedding_size) => (packed_sum_seq_len, hidden_size)
        out = self.dropout(seq.data)  # (packed_sum_seq_len, hidden_size) => (packed_sum_seq_len, hidden_size)
        out = self.linear(out)  # (packed_sum_seq_len, hidden_size) => (packed_sum_seq_len, output_size)
        return out


print("### Loading datas ###")
sequences, tags = read_data("train")
dev_sequences, dev_tags = read_data("dev")
print(f"total seqs: train:{len(sequences)} dev:{len(dev_sequences)}")

# 21class
i_to_tag, _ = zip(*Counter([tag for _tags in tags for tag in _tags]).most_common())
i_to_tag = list(i_to_tag)
tag_to_i = {t: i for i, t in enumerate(i_to_tag)}


hidden_size = 256
batch_size = 32
output_size = len(i_to_tag)
learning_rate = 0.001
epoch = 100

writer = SummaryWriter()
ner_dataloader = NERDataLoader(tag_to_i)

dataloader = ner_dataloader.get_dataloader("train", *(sequences, tags), shuffle=True, batch_size=batch_size)
dev_dataloader = ner_dataloader.get_dataloader("dev", *(dev_sequences, dev_tags), shuffle=False)  # Full batch

del sequences, tags, dev_sequences, dev_tags  # release memory

loss_fn = torch.nn.CrossEntropyLoss()
net = Net(ner_dataloader.embedding_size, hidden_size, output_size)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

print("### Start training ###")
for i in range(epoch):
    train_loss = []
    # train
    net.train()
    for padded_seqs, tags_target in dataloader:
        tags_predict = net(padded_seqs)
        loss = loss_fn(tags_predict, tags_target)  # CrossEntropy target is label index
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 0.5)  # prevent gradient explode
        optimizer.step()
    # dev
    net.eval()
    with torch.no_grad():
        padded_seqs, tags_target = next(iter(dev_dataloader))  # full batch
        tags_predict = net(padded_seqs)
        dev_loss = loss_fn(tags_predict, tags_target).item()
        _, _, f1, non_o_accuracy, accuracy = evaluate(
            [i_to_tag[idx] for idx in tags_target.tolist()],
            [i_to_tag[idx] for idx in tags_predict.argmax(dim=1).tolist()],
            True
        )
    # record the history
    train_loss = sum(train_loss) / len(train_loss)
    writer.add_scalar("train_loss", train_loss, i)
    writer.add_scalar("dev_loss", dev_loss, i)
    writer.add_scalar("dev_f1", f1, i)
    writer.add_scalar("dev_non_o_accuracy", non_o_accuracy, i)
    writer.add_scalar("dev_accuracy", accuracy, i)
    print(f"{i:02d}/{epoch}: train: {train_loss:.16f} test: {dev_loss:.16f}\n")
torch.save(net.state_dict(), "./submit.model")

del dataloader, dev_dataloader  # release memory

print("### Start testing ###")
submit_dataloader = ner_dataloader.get_dataloader("submit", *read_data("test-submit"), shuffle=False, batch_size=1024)

submit_text = []
net.eval()
with torch.no_grad():
    for padded_seqs, real_seqs in submit_dataloader:
        tags_predict = net(padded_seqs)
        packed = PackedSequence(data=tags_predict, batch_sizes=padded_seqs.batch_sizes, sorted_indices=padded_seqs.sorted_indices, unsorted_indices=padded_seqs.unsorted_indices)
        padded, out_len = pad_packed_sequence(packed, batch_first=True, padding_value=float("-inf"))  # (packed_sum_seq_len, output_size) => (batch_size, max_seq_len, output_size)
        padded = padded.argmax(dim=-1)  # (batch_size, max_seq_len, output_size) => (batch_size, max_seq_len)
        pred_tag_seqs = unpad_sequence(padded, out_len, batch_first=True)  # [batch_size, each_seq_len]
        for pred_tag_seq, real_seq in zip(pred_tag_seqs, real_seqs):  # each batch(seq)
            submit_text += [f"{real_word}\t{i_to_tag[pred_tag]}\n" for pred_tag, real_word in zip(pred_tag_seq, real_seq)] + ["\n"]
with Path("./my_submit.txt").open("w", encoding="utf-8") as f:
    f.writelines(submit_text)
print("### Submit file saved ###")
