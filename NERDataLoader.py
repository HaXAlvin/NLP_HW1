import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from NERDataSet import NERDataSet, create_pretrained_embedding


class NERDataLoader:
    def __init__(self, tag_to_i, twitter=False):
        if twitter:
            embedding, word_to_i = create_pretrained_embedding("twitter.27B", 200)
            self.embedding_size = 200
        else:
            embedding, word_to_i = create_pretrained_embedding("840B", 300)
            self.embedding_size = 300
        self.embedding = embedding
        self.word_to_i = word_to_i
        self.tag_to_i = tag_to_i

    def sequences_to_padded_index(self, submit):  # to create the callback function (collate_fn) for dataloader
        def _sequences_to_padded_index(batched_data):
            # batched data from dataloader to packed data, if submit then change tag to real word
            # seq => pad seq => pad emb => pack emb
            # tag => pad tag => pack tag
            # real => no change
            batch_seq_indexes = []
            batch_tag_indexes = []  # only use if submit=False
            batch_real_words = []  # only use if submit=True
            seqs_len = []
            unk = self.word_to_i["<unk>"]

            for seqs, tags_or_words in batched_data:
                seqs_len.append(len(seqs))
                batch_seq_indexes.append(torch.tensor([self.word_to_i.get(word, unk) for word in seqs]))
                if submit:
                    batch_real_words.append(tags_or_words)
                else:
                    batch_tag_indexes.append(torch.tensor([self.tag_to_i[tag] for tag in tags_or_words]))

            padded_seqs = pad_sequence(batch_seq_indexes, batch_first=True, padding_value=self.word_to_i["<pad>"])  # (batch_size, each_seq_len) => (batch_size, max_seq_len)
            packed_seqs = pack_padded_sequence(self.embedding(padded_seqs), seqs_len, batch_first=True, enforce_sorted=False)  # (batch_size, max_seq_len) => (batch_size, max_seq_len, embedding_size) => (packed_sum_seq_len, embedding_size)
            if submit:
                return packed_seqs, batch_real_words

            padded_tags = pad_sequence(batch_tag_indexes, batch_first=True, padding_value=-1)  # (batch_size, each_seq_len) => (batch_size, max_seq_len)
            packed_tags = pack_padded_sequence(padded_tags, seqs_len, batch_first=True, enforce_sorted=False)  # (batch_size, max_seq_len) => (packed_sum_seq_len)

            # pack will sort input by length, this is to make sure word-tag pair will not unpaired after pack
            assert packed_seqs.batch_sizes.tolist() == packed_tags.batch_sizes.tolist() and \
                packed_seqs.sorted_indices.tolist() == packed_tags.sorted_indices.tolist() and \
                packed_seqs.unsorted_indices.tolist() == packed_tags.unsorted_indices.tolist()
            return packed_seqs, packed_tags.data
        return _sequences_to_padded_index

    def get_dataloader(self, mode, *data, shuffle=False, batch_size=None) -> DataLoader:
        assert mode in ["train", "dev", "submit"]
        if batch_size is None:
            batch_size = len(data[0])
        return DataLoader(NERDataSet(*data), batch_size, shuffle, collate_fn=self.sequences_to_padded_index(mode == "submit"))
