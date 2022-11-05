from torch.utils.data import DataLoader
import torch
from NERDataSet import get_data, NERDataSet, create_pretrained_embedding
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from collections import Counter
from conlleval import evaluate
from torch.utils.tensorboard import SummaryWriter


def sequences_to_padded_index(submit=False): # to create the callback function (collate_fn) for dataloader
    def _sequences_to_padded_index(batched_data):
        # batched data from dataloader to packed data, if submit then change tag to real word
        # seq => pad seq => pad emb => pack emb
        # tag => pad tag => pack tag
        # real => no change
        batch_seq_indexs = []
        batch_tag_indexs = [] # only use if submit=False
        batch_real_words = [] # only use if submit=True
        seqs_len = []
        unk = word_to_i["<unk>"]

        for seqs, tags_or_words in batched_data:
            seqs_len.append(len(seqs))
            batch_seq_indexs.append(torch.tensor([word_to_i.get(word, unk) for word in seqs]))
            if submit:
                batch_real_words.append(tags_or_words)
            else:
                batch_tag_indexs.append(torch.tensor([tag_to_i[tag] for tag in tags_or_words]))

        padded_seqs = pad_sequence(batch_seq_indexs, batch_first=True, padding_value=word_to_i["<pad>"])
        packed_seqs = pack_padded_sequence(embedding(padded_seqs), seqs_len, batch_first=True, enforce_sorted=False)
        if submit:
            return packed_seqs, batch_real_words

        padded_tags = pad_sequence(batch_tag_indexs, batch_first=True, padding_value=-1)
        packed_tags = pack_padded_sequence(padded_tags, seqs_len, batch_first=True, enforce_sorted=False)

        # pack will sort input by length, this is to make sure word-tag pair will not unpaired after pack
        assert packed_seqs.batch_sizes.tolist() == packed_tags.batch_sizes.tolist() and \
               packed_seqs.sorted_indices.tolist() == packed_tags.sorted_indices.tolist() and \
               packed_seqs.unsorted_indices.tolist() == packed_tags.unsorted_indices.tolist()
        return packed_seqs, packed_tags.data
    return _sequences_to_padded_index


class Net(torch.nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, bidirectional=True, dropout=0.5):
        super(Net, self).__init__()
        self.gru = torch.nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout1 = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(2*hidden_size if bidirectional else hidden_size, output_size, bias=False)

    def forward(self, batch_seq):
        seq, _ = self.gru(batch_seq)
        out = self.dropout1(seq.data)
        out = self.linear(out)
        return out

print("### Loading datas ###")
sequences, tags = get_data()
dev_sequences, dev_tags = get_data("./datas/twitter/dev.txt")
print(f"total seqs: train:{len(sequences)} dev:{len(dev_sequences)}")

# 21class
i_to_tag, _ = zip(*Counter([tag for _tags in tags for tag in _tags]).most_common())
i_to_tag = list(i_to_tag)
tag_to_i = {t: i for i, t in enumerate(i_to_tag)}


embedding_size = 300
hidden_size = 256
batch_size = 32
output_size = len(i_to_tag)
learning_rate = 0.001
epoch = 100

writer = SummaryWriter()

embedding, word_to_i = create_pretrained_embedding("840B", embedding_size)  # 300
# embedding, word_to_i = create_pretrained_embedding("twitter.27B",embedding_size)#200

dataloader = DataLoader(NERDataSet(sequences, tags), batch_size, shuffle=True, collate_fn=sequences_to_padded_index())
dev_dataloader = DataLoader(NERDataSet(dev_sequences, dev_tags), batch_size=1024, collate_fn=sequences_to_padded_index()) # batch>1000 => full batch

loss_fn = torch.nn.CrossEntropyLoss()
net = Net(embedding_size, hidden_size, output_size)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

print("### Start training ###")
for i in range(epoch):
    train_loss = []
    dev_loss = []
    # train
    net.train()
    for padded_seqs, tags_target in dataloader:
        tags_predict = net(padded_seqs)
        loss = loss_fn(tags_predict, tags_target)  # CrossEntropy target is label index
        train_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), 0.5) # prevent gradient explode
        optimizer.step()
    # dev
    net.eval()
    with torch.no_grad():
        padded_seqs, tags_target = next(iter(dev_dataloader)) # full batch
        tags_predict = net(padded_seqs)
        loss = loss_fn(tags_predict, tags_target)
        dev_loss.append(loss.item())
        _, _, f1, non_o_accuracy, accuracy = evaluate(
            [i_to_tag[idx] for idx in tags_target.tolist()],
            [i_to_tag[idx] for idx in tags_predict.argmax(dim=1).tolist()],
            True
        )
    # record the history
    train_loss = sum(train_loss)/len(train_loss)
    dev_loss = sum(dev_loss)/len(dev_loss)
    writer.add_scalar("train_loss", train_loss, i)
    writer.add_scalar("dev_loss", dev_loss, i)
    writer.add_scalar("dev_f1", f1, i)
    writer.add_scalar("dev_non_o_accuracy", non_o_accuracy, i)
    writer.add_scalar("dev_accuracy", accuracy, i)
    print(f"{i:02d}/{epoch}: train: {train_loss:.16f} test: {dev_loss:.16f}\n")
torch.save(net.state_dict(), "./test.model")

submit_dataloader = DataLoader(NERDataSet(*get_data("./datas/twitter/test-submit.txt")), batch_size=1024, collate_fn=sequences_to_padded_index(True))

net.eval()
with torch.no_grad():
    with open("./my_submit.txt", "w", encoding="utf-8") as f:
        for padded_seqs, real_word in submit_dataloader:
            tags_predict = net(padded_seqs)
            packed = PackedSequence(data=tags_predict, batch_sizes=padded_seqs.batch_sizes, sorted_indices=padded_seqs.sorted_indices, unsorted_indices=padded_seqs.unsorted_indices)
            padded, out_len = pad_packed_sequence(packed, batch_first=True, padding_value=float("-inf"))
            padded = padded.argmax(dim=-1)
            for i, (real_seq, seq_len) in enumerate(zip(real_word, out_len)):
                for word, tag in zip(real_seq, padded[i][:seq_len].tolist()):
                    f.write(f"{word}\t{i_to_tag[tag]}\n")
                f.write("\n")
