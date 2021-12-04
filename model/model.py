from collections import OrderedDict
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
from utils.utils import compute_f1, compute_sentences_similar, create_distractor
import numpy as np


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class GenEERModel(pl.LightningModule):
    def __init__(self, tokenizer_name: str, model_name_or_path: str,
                input_format: str, oupt_format: str, max_input_len: int, max_oupt_len: int,
                generate_weight: float, f1_weight: float,
                pretrain_step: int, reinforce_step: int, 
                pretrain_lr: float, reinforce_lr: float, reconstructor_lr: float,
                adam_epsilon: float, weight_decay: float=0, warmup: float=0, gamma: float=0.9) -> None:
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
        # self.drop_out = nn.Dropout(0.9)
        # self.mlp = nn.Sequential(OrderedDict([
        #                                     ('dropout1',self.drop_out), 
        #                                     ('fc1', nn.Linear(2, 2)), 
        #                                     ('dropout2', self.drop_out), 
        #                                     ('relu', nn.ReLU()), 
        #                                     ('fc2', nn.Linear(2, 1)),]))
        # self.fn_actiavte = nn.Sigmoid()
        
        # self.rewards = []
        # self.val_rewards = []
        
        self.automatic_optimization = False
    
    def compute_reward(self, inputs_sentences, generated_output, golds):
        #-------------------F1_REWARD-----------------------
        f1_reward = compute_f1(generated_output, golds)[0]
        f1_reward = f1_reward * torch.ones((len(inputs_sentences))) 

        #---------------------RECONSTRUCT REWARD-----------------------
        # with torch.no_grad():
        #     task_prefix = 'Generate question and context'
        #     reconstruct_inputs_emcoding = self.tokenizer_for_generating([f"{task_prefix}:\n{sent}" for sent in generated_output],
        #                                                                     padding='longest',
        #                                                                     max_length=self.hparams.max_oupt_len,
        #                                                                     truncation=True,
        #                                                                     return_tensors="pt")
        #     reconstructed_inputs = self.t5.generate(input_ids=reconstruct_inputs_emcoding.input_ids.cuda(), 
        #                                     do_sample=False, 
        #                                     top_k=20, 
        #                                     top_p=0.95, 
        #                                     max_length=self.hparams.max_oupt_len, 
        #                                     num_return_sequences=1, 
        #                                     num_beams=1,)
        #     reconstructed_inputs = self.tokenizer_for_generating.batch_decode(reconstructed_inputs, skip_special_tokens=True)
        #     avg_sim = compute_sentences_similar(inputs_sentences, reconstructed_inputs)
        # reconstruct_reward = 0
        # print(f1_reward)
        # self.log_dict({'f1_reward': f1_reward, 'reconstruct_reward': reconstruct_reward}, prog_bar=True)
        return f1_reward

    def compute_generate_loss(self, inputs, outputs):
        # generate loss (in->out)
        task_prefix = 'Causality identification'
        inputs_encoding = self.tokenizer([f"{task_prefix}:\n{sent}" for sent in inputs], 
                                                    padding='longest',
                                                    max_length=self.hparams.max_input_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        
        outputs_encoding = self.tokenizer(outputs,
                                                padding='longest',
                                                max_length=self.hparams.max_oupt_len,
                                                truncation=True,
                                                return_tensors="pt")
        labels = outputs_encoding.input_ids
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        _generate_output = self.t5(
                                input_ids=inputs_encoding.input_ids.cuda(), 
                                attention_mask=inputs_encoding.attention_mask.cuda(), 
                                labels=labels.cuda(),
                                output_hidden_states=True
                    )
        generate_loss = _generate_output.loss
        # last_generate_hidden_state = _generate_output.decoder_hidden_states[-1] # (batch_size, sequence_length, hidden_size)
        # inputs_encoding_for_generating = self.tokenizer_for_generating([f"{task_prefix}:\n{sent}" for sent in inputs], 
        #                                                                     padding='longest',
        #                                                                     max_length=self.hparams.max_input_len,
        #                                                                     truncation=True,
        #                                                                     return_tensors="pt")
        # sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), 
        #                                     do_sample=True, 
        #                                     top_k=20, 
        #                                     top_p=0.95, 
        #                                     max_length=self.hparams.max_oupt_len, 
        #                                     num_return_sequences=1, 
        #                                     num_beams=1,)
        # sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

        return generate_loss, _generate_output, labels, 'sample_outputs'

    def compute_reconstruct_loss(self, inputs, outputs):
        # reconstruct loss (out->in)
        task_prefix = 'Generate question and context'
        reconstruct_inputs_embedding = self.tokenizer([f"{task_prefix}:\n{sent}" for sent in outputs],
                                                    padding='longest',
                                                    max_length=self.hparams.max_oupt_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        reconstruct_outputs_embedding = self.tokenizer(inputs, 
                                                    padding='longest',
                                                    max_length=self.hparams.max_input_len,
                                                    truncation=True,
                                                    return_tensors="pt")
        reconstruct_labels = reconstruct_outputs_embedding.input_ids
        reconstruct_labels[reconstruct_labels[:, :] == self.tokenizer.pad_token_id] = -100 # replace padding token id's of the labels by -100
        
        _reconstruct_output = self.t5(
                                input_ids=reconstruct_inputs_embedding.input_ids.cuda(), 
                                attention_mask=reconstruct_inputs_embedding.attention_mask.cuda(), 
                                labels=reconstruct_labels.cuda(),
                                output_hidden_states=True
                    )
        reconstruct_loss = _reconstruct_output.loss
        # last_reconstruct_hidden_state = _reconstruct_output.decoder_hidden_states[-1] # (batch_size, sequence_length, hidden_size)

        return reconstruct_loss, _reconstruct_output

    def warmup_step(self, batch):
        # generate score
        template = [5]*len(batch)
        inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                            for temp_id, example in zip(template, batch)]
        
        generate_loss, _, _, generated_ouput = self.compute_generate_loss(inputs=inputs_sentences, outputs=output_sentences) # in -> out

        reconstruct_loss = self.compute_reconstruct_loss(inputs=inputs_sentences, outputs=generated_ouput)[0] # out -> in 
        # self.log_dict({'pretrain_gen_loss': generate_loss, 'pretrain_reconstruct_loss': reconstruct_loss}, prog_bar=True)
        return self.hparams.generate_weight * generate_loss + (1.0 - self.hparams.generate_weight) * reconstruct_loss
    
    def reinforce_training_step(self, batch):
        template = [5]*len(batch)
        inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                            for temp_id, example in zip(template, batch)]
        _, generate_output, labels, _ = self.compute_generate_loss(inputs=inputs_sentences, outputs=output_sentences)
        
        logits = generate_output.logits
        # print(f"logit: {logits}")
        probs = torch.softmax(logits, dim=-1)
        # print(f"prob: {probs}")
        distribution = torch.distributions.Categorical(probs=probs)
        output_seqs = distribution.sample() # (batch_size, seq_len)
        # print(f"output_seqs: {output_seqs}")
        log_probs = distribution.log_prob(output_seqs) # (batch_size, seq_len)
        # print(log_probs)
        mask = torch.ones(labels.size()).cuda()
        mask[labels[:, :] == -100] = 0
        print(mask.size())
        print(log_probs.size())
        log_probs = log_probs * mask

        outputs = self.tokenizer.batch_decode(output_seqs, skip_special_tokens=True)
        reward = self.compute_reward(inputs_sentences, generated_output=outputs, golds=output_sentences)

        policy_gradients = 0
        for seq_id in range(log_probs.size(0)):
            seq_logprobs = log_probs[seq_id]
            seq_rewards = [reward[seq_id]] * seq_logprobs.size(0) # all step has same reward which is all sentence reward.
            discounted_rewards = []
            for t in range(len(seq_rewards)):
                Gt = 0
                pw = 0
                for r in seq_rewards[t:]:
                    Gt = Gt + self.hparams.gamma**pw * r
                    pw = pw + 1
                discounted_rewards.append(Gt)
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
            policy_gradient = []
            for log_prob, Gt in zip(seq_logprobs, discounted_rewards):
                policy_gradient.append(- log_prob * Gt)
            policy_gradient = torch.stack(policy_gradient).sum() 
            policy_gradients = policy_gradients + policy_gradient # sum all gradient in batch 

        return outputs, reward, log_probs, policy_gradients

    def on_train_epoch_start(self) -> None:
        self.rewards = []
    
    def training_step(self, batch, batch_idx):
        pretrain_optimizer, reinforce_optimizer = self.optimizers()
        pretrain_scheduler = self.lr_schedulers()
        # print(self.trainer.global_step)

        if self.trainer.global_step < self.hparams.pretrain_step:
            
            pretrain_loss = self.warmup_step(batch)
            pretrain_optimizer.zero_grad()
            self.manual_backward(pretrain_loss)
            pretrain_optimizer.step()
            pretrain_scheduler.step()
            self.log_dict({'pretrain_loss': pretrain_loss}, prog_bar=True)
        else:
            outputs, reward, log_probs, reinforce_loss = self.reinforce_training_step(batch, )
            self.rewards.append(reward)

            # template = [5]*len(batch)
            # inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
            #                         for temp_id, example in zip(template, batch)]
            # reconstruct_loss = self.compute_reconstruct_loss(inputs=inputs_sentences, outputs=outputs)[0]

            # reinforce_loss = 0.1 * reconstruct_loss + reinforce_loss
            # reconstructor_optimizer.zero_grad()
            # self.manual_backward(reconstruct_loss)
            # reconstructor_optimizer.step()
            # reconstructor_scheduler.step()
            # self.log_dict({'reconstruct_loss': reconstruct_loss})
            
            reinforce_optimizer.zero_grad()
            self.manual_backward(reinforce_loss)
            reinforce_optimizer.step()
            # reinforce_scheduler.step()
            self.log_dict({'reinforce_loss': reinforce_loss}, prog_bar=True)

    def on_validation_epoch_start(self) -> None:
        self.val_rewards = []
    
    def validation_step(self,batch, batch_idx):
        if self.trainer.global_step > self.hparams.pretrain_step:
            template = [5]*len(batch)
            task_prefix = 'Causality identification'
            inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                    for temp_id, example in zip(template, batch)]
            inputs_encoding_for_generating = self.tokenizer_for_generating([f"{task_prefix}:\n{sent}" for sent in inputs_sentences], 
                                                                            padding='longest',
                                                                            max_length=self.hparams.max_input_len,
                                                                            truncation=True,
                                                                            return_tensors="pt")
            
            # generate output 
            sample_outputs = self.t5.generate(input_ids=inputs_encoding_for_generating.input_ids.cuda(), 
                                            do_sample=False, 
                                            top_k=20, 
                                            top_p=0.95, 
                                            max_length=self.hparams.max_oupt_len, 
                                            num_return_sequences=1, 
                                            num_beams=1,)
            sample_outputs = self.tokenizer_for_generating.batch_decode(sample_outputs, skip_special_tokens=True)

            # gold output
            output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                                for temp_id, example in zip(template, batch)]
            return sample_outputs, output_sentences, [f"{task_prefix}:\n{sent}" for sent in inputs_sentences]
    
    def validation_epoch_end(self, outputs):
        if self.trainer.global_step > self.hparams.pretrain_step:
            # evaluate F1
            golds = []
            predicts = []
            preds = []
            for output in outputs:
                for sample in zip(*output):
                    predicts.append(sample[0])
                    golds.append(sample[1])
                    preds.append({
                        'sentence': sample[2],
                        'predicted': sample[0],
                        'gold': sample[1]
                    })
            f1, p, r, tp, n_pred, n_gold = compute_f1(predicts, golds)
            # print("DEV result:")
            # print(f"f1: {f1}")
            with open('./reinforce_model_predictions_dev.json','w') as writer:
                writer.write(json.dumps(preds, indent=6)+'\n')
            self.log_dict({'f1_dev': f1}, prog_bar=True)
            return f1
    
    def test_step(self, batch, batch_idx):
        template = [5]*len(batch)
        task_prefix = 'Causality identification'
        inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        inputs_encoding_for_generating = self.tokenizer_for_generating([f"{task_prefix}:\n{sent}" for sent in inputs_sentences], 
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
        return sample_outputs, output_sentences, [f"{task_prefix}:\n{sent}" for sent in inputs_sentences]

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

        with open('./reinforce_model_predictions.json','w') as writer:
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
        # reinforce_scheduler = get_linear_schedule_with_warmup(
        #     reinforce_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.hparams.reinforce_step
        # )

        # # config optimizer for reconstructor 
        # optimizer_grouped_recontructor_parameters = [
        #     {
        #         "params": [p for n, p in self.t5.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.hparams.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in self.t5.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # reconstructor_optimizer = AdamW(optimizer_grouped_recontructor_parameters, lr=self.hparams.reconstructor_lr, eps=self.hparams.adam_epsilon)
        # num_warmup_steps = self.hparams.warmup * self.hparams.reinforce_step
        # reconstructor_scheduler = get_linear_schedule_with_warmup(
        #     reconstructor_optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.hparams.reinforce_step
        # )

        return ({
            "optimizer": pretrain_optimizer,
            "lr_scheduler": pretrain_scheduler,
        },
            {
            "optimizer": reinforce_optimizer,
            # "lr_scheduler": reinforce_scheduler,
        },
        #     {
        #     "optimizer": reconstructor_optimizer,
        #     "lr_scheduler": reconstructor_scheduler,
        # }
        )
