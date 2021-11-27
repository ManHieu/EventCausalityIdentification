from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Tokenizer
from data_modules.input_formats import INPUT_FORMATS
from model.predictor_model import T5WithGenerateForReinforce


class AdditionalInfoGenerator(nn.Module):
    default_input_format = 'plain'

    def __init__(self, generator: T5WithGenerateForReinforce,
                tokenizer: T5Tokenizer, max_input_len: int, max_output_len: int,
                input_format: str=None,):
        super().__init__()

        self.input_format = INPUT_FORMATS[input_format]() if input_format != None else INPUT_FORMATS[self.default_input_format]()
        self.tokenizer = tokenizer

        self.generator = generator
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def forward(self, examples):
        task_prefix = 'generate additional information'
        input_sentence = [self.input_format.format_input_for_selector(example=example, task_prefix=task_prefix) 
                         for example in examples]
        inputs_encoding_for_additional_info_generator = self.tokenizer(input_sentence, 
                                                                    padding='longest',
                                                                    max_length=self.max_input_len,
                                                                    truncation=True,
                                                                    return_tensors="pt")
        
        # last_hidden_state  = self.sentence_encoder(
        #                 input_ids=inputs_encoding_for_template_selector.input_ids.cuda(),
        #                 attention_mask=inputs_encoding_for_template_selector.attention_mask.cuda(),
        #                 output_hidden_states=True,
        # ).last_hidden_state  # (batch_size, sequence_length, hidden_size)

        # sc = self.s_classifier(last_hidden_state[:, 0]) # (batch_size x num_templates)
        
        # probs = F.softmax(sc, dim=-1)
        # if self.training:
        #     action_distribution = torch.distributions.Categorical(probs)
        #     action = action_distribution.sample() # bs x 1: index of selected template 
        #     log_probs = action_distribution.log_prob(action) # bs x 1: log_prob of actions
        # else:
        #     action = probs.max(dim=-1)[1]
        #     log_probs = torch.log(probs.max(dim=-1)[0])


        # return action, log_probs, probs
        self.generator.config.max_length = self.max_output_len
        generated_additional_info = self.generator.generate_for_reinforce(
                                                                        input_ids=inputs_encoding_for_additional_info_generator.input_ids.cuda(), 
                                                                        do_sample=True, 
                                                                        top_k=20, top_p=0.95, 
                                                                        # max_length=self.max_output_len, 
                                                                        num_return_sequences=1, num_beams=1,
                                                                        return_dict_in_generate=True,
                                                                        output_scores=True
                                                                        )

        # print(f"generated_additional_info: {generated_additional_info}")
        ouput_seqs, scores = generated_additional_info.sequences, generated_additional_info.scores
        # print(len(scores))
        # print(ouput_seqs.size())
        log_probs = []
        for batch_id in range(ouput_seqs.size(0)):
            seq = ouput_seqs[batch_id]
            log_prob = []
            for i, tok in enumerate(seq):
                if tok not in [0, 1]:
                    score_in_step = scores[i-1]
                    logit_score = F.softmax(score_in_step, dim=1)
                    # print(score_in_step[batch_id][tok])
                    log_prob.append(logit_score[batch_id][tok])
            log_probs.append(log_prob)
        
        ouput_seqs = self.tokenizer.batch_decode(ouput_seqs, skip_special_tokens=True)
        # print(ouput_seqs)
        
        return ouput_seqs, log_probs
        
        # print(log_probs)

        # print(f"ouput_seq: {ouput_seq}")
        # print(f"score: {score}")
        



