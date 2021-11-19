from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer
from data_modules.input_formats import INPUT_FORMATS


class Selector(nn.Module):
    default_input_format = 'plain'

    def __init__(self, sentence_encoder: str, number_layers: int,
                tokenizer: T5Tokenizer, max_input_len: int,
                input_format: str=None, fn_activate: str = 'leakyrelu',):
        super().__init__()

        self.input_format = INPUT_FORMATS[input_format]() if input_format != None else INPUT_FORMATS[self.default_input_format]()
        self.tokenizer = tokenizer

        self.sentence_encoder = T5EncoderModel.from_pretrained(sentence_encoder)
        self.max_input_len = max_input_len

        self.hidden_dim = 768
        self.number_layers = number_layers

        self.dropout = nn.Dropout(0.5)

        if fn_activate=='leakyrelu':
            self.fn_activate = nn.LeakyReLU(0.2, True)
        elif fn_activate=='tanh':
            self.fn_activate = nn.Tanh()
        elif fn_activate=='relu6':
            self.fn_activate = nn.ReLU6()
        elif fn_activate=='silu':
            self.fn_activate = nn.SiLU()
        elif fn_activate=='hardtanh':
            self.fn_activate = nn.Hardtanh()

        self.s_classifier = nn.Sequential(OrderedDict([
                                                    ('dropout1', self.dropout),
                                                    ('mlp1', nn.Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim/2))),
                                                    ('dropout2', self.dropout),
                                                    ('activate', self.fn_activate),
                                                    ('mlp2', nn.Linear(in_features=int(self.hidden_dim/2), out_features=self.number_layers)) # num_templates temporaly fixed is 2
        ]))

    def forward(self, input_sentence):
        task_prefix = 'select input template'
        input_sentence = [self.input_format.format_input_for_selector(ctx=sentence, task_prefix=task_prefix) 
                         for sentence in input_sentence]
        inputs_encoding_for_template_selector = self.tokenizer(input_sentence, 
                                                                padding='longest',
                                                                max_length=self.max_input_len,
                                                                truncation=True,
                                                                return_tensors="pt")
        
        last_hidden_state  = self.sentence_encoder(
                        input_ids=inputs_encoding_for_template_selector.input_ids.cuda(),
                        attention_mask=inputs_encoding_for_template_selector.attention_mask.cuda(),
                        output_hidden_states=True,
        ).last_hidden_state  # (batch_size, sequence_length, hidden_size)

        sc = self.s_classifier(last_hidden_state[:, 0]) # (batch_size x num_templates)
        
        probs = F.softmax(sc, dim=-1)
        if self.training:
            action_distribution = torch.distributions.Categorical(probs)
            action = action_distribution.sample() # bs x 1: index of selected template 
            log_probs = action_distribution.log_prob(action) # bs x 1: log_prob of actions
        else:
            action = probs.max(dim=-1)[1]
            log_probs = torch.log(probs.max(dim=-1)[0])

        return action, log_probs, probs


