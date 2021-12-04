from typing import List, Optional
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import T5Tokenizer
from T5ForRL import T5ForRL
from transformers import AdamW, get_linear_schedule_with_warmup
import copy

from data_modules.input_formats import INPUT_FORMATS
from data_modules.output_formats import OUTPUT_FORMATS
from utils.utils import compute_f1


class GenEC(pl.LightningModule):
    def __init__(self,
                tokenizer_name: str,
                model_name_or_path: str,
                input_type: str,
                output_type: str,
                max_input_len: int,
                max_oupt_len: int,
                num_training_step: int,
                lr: float,
                warmup: float,
                adam_epsilon: float,
                weight_decay: float,
                generate_weight: float,
                ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.t5: T5ForRL = T5ForRL.from_pretrained(model_name_or_path)

        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        
        self.tokenizer_for_generate: T5Tokenizer = copy.deepcopy(self.tokenizer)
        self.tokenizer_for_generate.padding_side = 'left'
        self.tokenizer_for_generate.pad_token = self.tokenizer_for_generate.eos_token

        self.input_formater = INPUT_FORMATS[input_type]()
        self.oupt_formater = OUTPUT_FORMATS[output_type]()
    
    def _forward(self, inputs: List[str], outputs: List[str], task_prefix: str):
        """
        """
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

        output = self.t5(
                        input_ids=inputs_encoding.input_ids.cuda(), 
                        attention_mask=inputs_encoding.attention_mask.cuda(), 
                        labels=labels.cuda(),
                        output_hidden_states=True)
        
        return output, labels
    
    def _generate(self, inputs: List[str], task_prefix: str):
        """
        """
        inputs_encoding_for_generate = self.tokenizer_for_generate([f"{task_prefix}:\n{sent}" for sent in inputs],
                                                                    padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,
                                                                    return_tensors="pt")
        generated_seqs = self.t5.generate(input_ids=inputs_encoding_for_generate.input_ids.cuda(), 
                                        do_sample=True, 
                                        top_k=20, 
                                        top_p=0.95, 
                                        max_length=self.hparams.max_oupt_len, 
                                        num_return_sequences=1, 
                                        num_beams=1,)
        generated_seqs = self.tokenizer_for_generate.batch_decode(generated_seqs, skip_special_tokens=True)
            
        return generated_seqs

    def train_MLE(self, batch):
        template = [5]*len(batch)
        gold_inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        gold_output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                                for temp_id, example in zip(template, batch)]

        # generate answer
        task_prefix = 'Identify causality relation'
        output_of_generating, _ = self._forward(inputs=gold_inputs_sentences, outputs=gold_output_sentences, task_prefix=task_prefix)
        generated_seqs = self._generate(gold_inputs_sentences, task_prefix)
        generate_loss = output_of_generating.loss

        # reconstruct question
        task_prefix = 'Generate question and context'
        output_of_reconstructing, _ = self._forward(inputs=generated_seqs, outputs=gold_inputs_sentences, task_prefix=task_prefix)
        reconstruct_loss = output_of_reconstructing.loss

        mle_loss = self.hparams.generate_weight * generate_loss + (1.0 - self.hparams.generate_weight) * reconstruct_loss
        
        return mle_loss

    def train_RL(self, batch):
        template = [5]*len(batch)
        gold_inputs_sentences = [self.input_formater.format_input(example=example, template_type=temp_id)[-1]
                                for temp_id, example in zip(template, batch)]
        gold_output_sentences = [self.oupt_formater.format_output(example=example, template_type=temp_id)
                                for temp_id, example in zip(template, batch)]

        # compute log_probs, get sample ouputs
        task_prefix = 'Identify causality relation'
        inputs_encoding_for_generate = self.tokenizer_for_generate([f"{task_prefix}:\n{sent}" for sent in gold_inputs_sentences],
                                                                    padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,
                                                                    return_tensors="pt")
        # generate seqence using sample 
        sampled_seqs = self.t5.generate_for_rl_training(input_ids=inputs_encoding_for_generate.input_ids.cuda(), 
                                                        do_sample=True, 
                                                        top_k=20, 
                                                        top_p=0.95, 
                                                        max_length=self.hparams.max_oupt_len, 
                                                        num_return_sequences=1, 
                                                        num_beams=1,
                                                        return_dict_in_generate=True)
        logits = 
        
        pass
    
    def compute_reward(generated_sents, origin_sents):
        #-------------------F1_REWARD-----------------------
        f1_reward = compute_f1(generated_sents, origin_sents)
        f1_reward = [f1_reward]*len(generated_sents)
        f1_reward = torch.tensor(f1_reward, dtype=torch.float)
        
        reward = f1_reward.cuda()
        return reward

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pass

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        pass

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_pretrain_parameters = [
            {
                "params": [p for n, p in self.t5.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.t5.named_parameters() if  any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },]
        optimizer = AdamW(optimizer_grouped_pretrain_parameters, lr=self.hparams.lr, eps=self.hparams.adam_epsilon)
        num_warmup_steps = self.hparams.warmup * self.hparams.num_training_step
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.hparams.num_training_step
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step'
            }
        }

