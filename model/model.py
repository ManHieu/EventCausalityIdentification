import copy
from typing import Dict, List
from statistics import mean
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import logging 
import json 
import pytorch_lightning as pl
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from data_modules.input_example import InputExample, Relation, RelationType
from data_modules.input_formats import INPUT_FORMATS
from data_modules.output_formats import OUTPUT_FORMATS
from utils.utils import compute_f1
import numpy as np


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class GenEERModel(pl.LightningModule):
    def __init__(self, tokenizer_name: str, model_name_or_path: str,
                input_format: str, oupt_format: str, max_input_len: int, max_oupt_len: int,
                generate_weight: float, f1_weight: float, margin: float,
                pretrain_step: int, reinforce_step: int, pretrain_lr: float, reinforce_lr: float, 
                adam_epsilon: float, weight_decay: float=0, warmup: float=0) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_for_generating = copy.deepcopy(self.tokenizer)
        # when generating, we will use the logits of right-most token to predict the next token
        # so the padding should be on the left
        self.tokenizer_for_generating.padding_side = 'left'
        self.tokenizer_for_generating.pad_token = self.tokenizer_for_generating.eos_token # to avoid an error
        
        self.input_formater = INPUT_FORMATS[input_format]()
        self.oupt_formater = OUTPUT_FORMATS[oupt_format]()
        self.number_templates = len(self.input_formater.templates) - 1

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.mlp = nn.Linear(2, 1, bias=True)
        self.fn_actiavte = nn.Sigmoid()
        
        self.rewards = []
        self.val_rewards = []
        
        self.automatic_optimization = False
    
    def compute_reward(self, inputs_sentences, generated_output, golds):
        #-------------------F1_REWARD-----------------------
        f1_reward = compute_f1(generated_output, golds)[0]

        #---------------------RECONSTRUCT REWARD-----------------------
        with torch.no_grad():
            score, _ = self.scoring_in_out(inputs_sentences, generated_output)
            random.shuffle(inputs_sentences)
            distractor_score, _ = self.scoring_in_out(inputs_sentences, generated_output)
            
            reconstruct_reward = - max(0, self.hparams.margin + float(distractor_score) - float(score))
        self.log_dict({'f1': f1_reward, 'reconstruct': reconstruct_reward}, prog_bar=True)
        return float(self.hparams.f1_weight * f1_reward + (1 - self.hparams.f1_weight) * reconstruct_reward) 

    def scoring_in_out(self, inputs_sentences, output_sentences):
        # generate loss (in->out)
        task_prefix = 'Causality identification'
        inputs_sentences_encoding = self.tokenizer([f"{task_prefix}\n\n{sent}" for sent in inputs_sentences], 
                                                    padding='longest',
                                                    max_length=self.hparams.max_input_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        
        output_sentence_encoding = self.tokenizer(output_sentences,
                                                padding='longest',
                                                max_length=self.hparams.max_oupt_len,
                                                truncation=True,
                                                return_tensors="pt")
        labels = output_sentence_encoding.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        generate_loss = self.t5(
                                input_ids=inputs_sentences_encoding.input_ids.cuda(), 
                                attention_mask=inputs_sentences_encoding.attention_mask.cuda(), 
                                labels=labels.cuda()
                    ).loss
        
        # reconstruct loss (out->in)
        task_prefix = 'Generate question and context'
        reconstruct_input_embedding = self.tokenizer([f"{task_prefix}\n\n{sent}" for sent in output_sentences],
                                                    padding='longest',
                                                    max_length=self.hparams.max_oupt_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        reconstruct_output_embedding = self.tokenizer(inputs_sentences, 
                                                    padding='longest',
                                                    max_length=self.hparams.max_input_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        
        reconstruct_labels = reconstruct_output_embedding.input_ids
        reconstruct_labels[reconstruct_labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        reconstruct_loss = self.t5(
                                input_ids=reconstruct_input_embedding.input_ids.cuda(), 
                                attention_mask=reconstruct_input_embedding.attention_mask.cuda(), 
                                labels=reconstruct_labels.cuda()
                    ).loss
        
        # print(generate_loss, reconstruct_loss)
        # print(torch.cat([generate_loss.unsqueeze(0), reconstruct_loss.unsqueeze(0)], dim=0))
        score = self.mlp(torch.cat([generate_loss.unsqueeze(0), reconstruct_loss.unsqueeze(0)], dim=0))
        # print(score)
        score = self.fn_actiavte(score)
        # print(score)
        return score, generate_loss

    def pretraining_step(self, batch):
        # generate score
        template = [5]*len(batch)
        inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                            for temp_id, example in zip(template, batch)]
        
        score, generate_loss = self.scoring_in_out(inputs_sentences, output_sentences)
        
        
        # distractor score
        random.shuffle(output_sentences)
        distractor_score, _ = self.scoring_in_out(inputs_sentences, output_sentences)
        
        # discriminative loss 
        comp_loss = torch.log(1 + torch.exp(distractor_score - score))

        return self.hparams.generate_weight * generate_loss + (1.0 - self.hparams.generate_weight) * comp_loss
    
    def reinforce_training_step(self, batch):
        template = [5]*len(batch)
        task_prefix = 'Causality identification'
        inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        inputs_sentences_encoding = self.tokenizer([f"{task_prefix}\n\n{sent}" for sent in inputs_sentences], 
                                                    padding='longest',
                                                    max_length=self.hparams.max_input_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                            for temp_id, example in zip(template, batch)]
        output_sentence_encoding = self.tokenizer(output_sentences,
                                                padding='longest',
                                                max_length=self.hparams.max_oupt_len,
                                                truncation=True,
                                                return_tensors="pt")
        labels = output_sentence_encoding.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        logits = self.t5(
                        input_ids=inputs_sentences_encoding.input_ids.cuda(), 
                        attention_mask=inputs_sentences_encoding.attention_mask.cuda(), 
                        labels=labels.cuda()
            ).logits # (batch_size, seq_len, vocab_size)
        
        probs = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=probs)
        output_seqs = distribution.sample() # (batch_size, seq_len)
        # print(f"output_seqs: {output_seqs}")
        log_probs = distribution.log_prob(output_seqs) # (batch_size, seq_len)
        
        outputs = self.tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
        # print(f"outputs: {outputs}")
        reward = self.compute_reward(inputs_sentences, generated_output=outputs, golds=output_sentences)

        return reward, log_probs

    def on_train_epoch_start(self) -> None:
        self.rewards = []
    
    def training_step(self, batch, batch_idx):
        pretrain_optimizer, reinforce_optimizer = self.optimizers()
        pretrain_scheduler, reinforce_scheduler = self.lr_schedulers()
        # print(self.trainer.global_step)

        if self.trainer.global_step < self.hparams.pretrain_step:
            pretrain_loss = self.pretraining_step(batch)
            
            pretrain_optimizer.zero_grad()
            self.manual_backward(pretrain_loss)
            pretrain_optimizer.step()
            pretrain_scheduler.step()
            self.log_dict({'pretrain_loss': pretrain_loss}, prog_bar=True)
        else:
            reward, log_probs = self.reinforce_training_step(batch)
            self.rewards.append(reward)
            normalized_reward = reward - mean(self.rewards)
            # print(f"Reward: {normalized_reward} - log_prob: {log_probs}")
            reinforce_loss = []
            for i in range(log_probs.size(0)):
                log_prob = torch.mean(log_probs[i]).unsqueeze(0) # normalize (avoid the case of different sentence lenth)
                reinforce_loss.append(-log_prob * normalized_reward)
            reinforce_loss = torch.cat(reinforce_loss).sum()

            reinforce_optimizer.zero_grad()
            self.manual_backward(reinforce_loss)
            reinforce_optimizer.step()
            reinforce_scheduler.step()
            self.log_dict({'reinforce_loss': reinforce_loss}, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.val_rewards = []
    
    def validation_step(self,batch, batch_idx):
        if self.trainer.global_step > self.hparams.pretrain_step:
            reward, log_probs = self.reinforce_training_step(batch)
            self.val_rewards.append(reward)
            normalized_reward = reward - mean(self.val_rewards)
            # print(f"Reward: {normalized_reward} - log_prob: {log_probs}")
            reinforce_loss = []
            for i in range(log_probs.size(0)):
                log_prob = torch.mean(log_probs[i]) # normalize (avoid the case of different sentence lenth)
                reinforce_loss.append(-log_prob * normalized_reward)
            reinforce_loss = torch.cat(reinforce_loss).sum()
            self.log_dict({"reinforce_val_loss": reinforce_loss}, prog_bar=True)
            return reinforce_loss

    
    def test_step(self, batch, batch_idx):
        template = [5]*len(batch)
        task_prefix = 'Causality identification'
        inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        inputs_encoding_for_generating = self.tokenizer_for_generating([f"{task_prefix}\n\n{sent}" for sent in inputs_sentences], 
                                                                        padding='longest',
                                                                        max_length=self.hparams.max_input_len,
                                                                        truncation=True,
                                                                        return_tensors="pt")
        
        # generate output 
        sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), 
                                        do_sample=True, 
                                        top_k=20, 
                                        top_p=0.95, 
                                        max_length=self.hparams.max_oupt_len, 
                                        num_return_sequences=1, 
                                        num_beams=8,)
        sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

        # gold output
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                             for temp_id, example in zip(template, batch)]

        return sample_outputs, output_sentences, [f"{task_prefix}\n\n{sent}" for sent in inputs_sentences]

    def test_epoch_end(self, outputs):
        # evaluate F1
        preds = []
        for output in outputs:
            for sample in zip(*output):
                preds.append({
                    'sentence': sample[2],
                    'predicted': sample[0],
                    'gold': sample[1]
                })

        with open('./predictions.json','w') as writer:
            writer.write(json.dumps(preds, indent=6)+'\n')

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        # config optimizer for pretrain steps
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_pretrain_parameters = [
            {
                "params": [p for n, p in self.t5.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.t5.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.mlp.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.mlp.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        pretrain_optimizer = AdamW(optimizer_grouped_pretrain_parameters, lr=self.hparams.pretrain_lr, eps=self.hparams.adam_epsilon)
        num_warmup_steps = self.hparams.warmup * self.hparams.pretrain_step
        pretrain_scheduler = get_linear_schedule_with_warmup(
            pretrain_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.hparams.pretrain_step
        )

        # config optimizer for reinforce step 
        optimizer_grouped_reinforce_parameters = [
            {
                "params": [p for n, p in self.t5.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.t5.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        reinforce_optimizer = AdamW(optimizer_grouped_reinforce_parameters, lr=self.hparams.reinforce_lr, eps=self.hparams.adam_epsilon)
        num_warmup_steps = self.hparams.warmup * self.hparams.reinforce_step
        reinforce_scheduler = get_linear_schedule_with_warmup(
            reinforce_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.hparams.reinforce_step
        )

        return ({
            "optimizer": pretrain_optimizer,
            "lr_scheduler": pretrain_scheduler,
        },
            {
            "optimizer": reinforce_optimizer,
            "lr_scheduler": reinforce_scheduler,
        })
