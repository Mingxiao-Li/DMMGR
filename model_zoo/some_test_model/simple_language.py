import torch.nn as nn
import torch 
import os 
import json 
from x.common.util import get_numpy_word_embed
from x import * 
from model_zoo.graph_models.graph_model_zoo.graph_modules import LanguageEncoder
from model_zoo.graph_models.graph_dataset import GraphDataset
from torch.utils.data import DataLoader, Dataset
import torch.optim as Optim 


class LanguagePrediction(nn.Module):

    def __init__(self,config):
        super(LanguageEncoder,self).__init__()

        if not config.use_glove_emb:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embeded_size)
        elif config.use_glove_emb:
            word2index_path = os.path.join(config.parent_path, config.word2index_path)
            assert os.path.exists(word2index_path)

            with open(word2index_path, "r") as f:
                word2index = json.load(f)

            pretrained_word_emb = get_numpy_word_embed(
                word2index,
                os.path.join(config.parent_path, config.pretrained_word_path),
            )

            num_words, word_dim = pretrained_word_emb.shape

            self.word_embedding = nn.Embedding(num_words, word_dim)
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_word_emb))
            self.word_embedding.weight.requires_grad = config.fine_tune
        
        self.language_encoder = LanguageEncoder(config)
        
        self.prediction_mid = nn.Linear(config.hidden_size, config.hidden_size)
        self.prediction = nn.Linear(config.hidden_size,2)
    

    def forward(self, item):
        question = item["question"]
        question_mask = item["question_mask"]
        seq_word_emb = self.word_embedding(question)
        seq_len = torch.sum(question_mask == 1, dim = 1)
        _, _, (ques_vec,_) = self.language_encoder(seq_word_emb, seq_len)

        prediction = self.prediction(torch.tanh(self.prediction_mid(ques_vec)))
        
        return prediction





if __name__ == "__main__":
    
    config_path = "experiment_configs/simple_language_config"
    _C = XCfgs(config_path)
    _C.proc()
    config = _C.get_config()
    language_model = LanguagePrediction(config)

    language_model.cuda()
    
    loss_fn = nn.CrossEntropyLoss(reduction="mean").cuda()
    
    optimizer = Optim.Adam(
            filter(lambda p: p.requires_grad, language_model.parameters()),
            lr=config.OPTIM.lr_base,
            betas=config.OPTIM.opt_betas,
            eps=config.OPTIM.opt_eps,
        )
    
    data_set = GraphDataset(config.DATA)
    data_loader = DataLoader(
        dataset = data_set,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = 4,
        pin_memory = False,
        drop_last = True
    )

