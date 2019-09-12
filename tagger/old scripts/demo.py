#!/usr/bin/python3
import torch
import pickle
import numpy as np

class DemoTagger:
    class Tagger(torch.nn.Module):
        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)

            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = torch.nn.Linear(hidden_dim, tagset_size)

        def forward(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
            return tag_scores

    def __init__(self):
        # load the 4 index dictionaries (2 for tokens and 2 tags) for POS and SEM
        self.pos_wd2id, self.pos_id2wd, self.pos_tg2id, self.pos_id2tg = pickle.load(open('pos_dicts.pkl', 'rb'))
        self.sem_wd2id, self.sem_id2wd, self.sem_tg2id, self.sem_id2tg = pickle.load(open('sem_dicts.pkl', 'rb'))
        # prepare and load the model
        self.EMBEDDING_DIM = 100
        self.HIDDEN_DIM = 5
        #load POS model
        self.pos_model = DemoTagger.Tagger(self.EMBEDDING_DIM, self.HIDDEN_DIM, len(self.pos_wd2id), len(self.pos_tg2id))
        self.pos_model.load_state_dict(torch.load('pos_model.pth'))
        self.pos_model.eval()
        #load SEM model
        self.sem_model = DemoTagger.Tagger(self.EMBEDDING_DIM, self.HIDDEN_DIM, len(self.sem_wd2id), len(self.sem_tg2id))
        self.sem_model.load_state_dict(torch.load('sem_model.pth'))
        self.sem_model.eval()

    def tag(self, s):
        #text = input(f'Os gwelwch yn dda, nodwch y testun ar gyfer tagio POS...\n[Please, enter text for POS-tagging]:\n').split()
        text = s.split()
        with torch.no_grad():
            pos_inputs = self._prepare_sequence(text, self.pos_wd2id)
            sem_inputs = self._prepare_sequence(text, self.pos_wd2id)
            pos_tags = self._score_to_tag(self.pos_model(pos_inputs), self.pos_id2tg)
            sem_tags = self._score_to_tag(self.sem_model(pos_inputs), self.sem_id2tg)
        return tuple(zip(text, pos_tags, sem_tags))

    ### private
    # convert the sequencies to indexes and tensors
    def _prepare_sequence(self, seq, to_ix):
        idxs = [to_ix.get(w,0) for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    # reconvert the score tensors to tags with index dictionaries
    def _score_to_tag(self, tag_scores, i_to_tag):
        tagged = []
        for preds in tag_scores:
            preds = list(np.array(preds))
            idx = preds.index(max(preds))
            tagged.append(i_to_tag[idx])
        return tagged


if __name__ == "__main__":
    from demo import DemoTagger
    sentence = "Rhywbeth ffraeth yn Gymraeg am gath a physgod neu rywbeth sut ddylwn i wybod ?"
    tagr = DemoTagger()
    result = tagr.tag(sentence)
    print(result)
    print(" ".join(f"{wd}/{pt}/{st}" for wd,pt,st in result))
