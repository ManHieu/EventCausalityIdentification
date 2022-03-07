import json
from typing import List, Optional, Sequence
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from transformers import T5Tokenizer
from model.T5ForRL import T5ForRL
from transformers import AdamW, get_linear_schedule_with_warmup
import copy
from sentence_transformers import SentenceTransformer, util
from data_modules.input_formats import INPUT_FORMATS
from data_modules.output_formats import OUTPUT_FORMATS
from utils.utils import compute_f1, compute_sentences_similar


class GenEC(pl.LightningModule):
    def __init__(self,
                tokenizer_name: str,
                model_name_or_path: str,
                input_type: str,
                output_type: str,
                max_input_len: int,
                max_oupt_len: int,
                mle_train: bool,
                rl_train: bool,
                num_training_step: int,
                lr: float,
                warmup: float,
                adam_epsilon: float,
                weight_decay: float,
                generate_weight: float,
                f1_reward_weight: float,
                reconstruct_reward_weight: float,
                mle_weight: float
                ) -> None:
        super().__init__()
        self.save_hyperparameters()

        print(f"Loading pretrain model from: {model_name_or_path}")
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
        inputs_encoding = self.tokenizer([f"{task_prefix}\n{sent}" for sent in inputs],
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
    
    @torch.no_grad()
    def _generate(self, inputs: List[str], task_prefix: str, num_beams: int=1, do_sample: bool=True):
        """
        """
        inputs_encoding_for_generate = self.tokenizer_for_generate([f"{task_prefix}\n{sent}" for sent in inputs],
                                                                    padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,
                                                                    return_tensors="pt")
        generated_seqs = self.t5.generate(input_ids=inputs_encoding_for_generate.input_ids.cuda(), 
                                        do_sample=do_sample, 
                                        top_k=20, 
                                        top_p=0.95, 
                                        max_length=self.hparams.max_oupt_len, 
                                        num_return_sequences=1, 
                                        num_beams=num_beams,)
        generated_seqs = self.tokenizer_for_generate.batch_decode(generated_seqs, skip_special_tokens=True)
            
        return generated_seqs

    def train_MLE(self, batch):
        gold_inputs_sentences = [self.input_formater.format_input(example=example)[-1] for example in batch]
        gold_output_sentences = [self.oupt_formater.format_output(example=example) for example in batch]

        # generate answer
        task_prefix = 'causality identification'
        output_of_generating, _ = self._forward(inputs=gold_inputs_sentences, outputs=gold_output_sentences, task_prefix=task_prefix)
        generate_loss = output_of_generating.loss

        reconstruct_loss = 0.0
        if self.hparams.rl_train==False:
            # reconstruct question
            generated_seqs = self._generate(gold_inputs_sentences, task_prefix)
            task_prefix = 'Generate question and context'
            output_of_reconstructing, _ = self._forward(inputs=generated_seqs, outputs=gold_inputs_sentences, task_prefix=task_prefix)
            reconstruct_loss = output_of_reconstructing.loss

        mle_loss = self.hparams.generate_weight * generate_loss + (1.0 - self.hparams.generate_weight) * reconstruct_loss
        
        return mle_loss

    def train_RL(self, batch):
        gold_inputs_sentences = [self.input_formater.format_input(example=example)[-1] for example in batch]
        gold_output_sentences = [self.oupt_formater.format_output(example=example) for example in batch]

        # compute log_probs, get sample ouputs
        task_prefix = 'causality identification'
        inputs_encoding_for_generate = self.tokenizer_for_generate([f"{task_prefix}\n{sent}" for sent in gold_inputs_sentences],
                                                                    padding='longest',
                                                                    max_length=self.hparams.max_input_len,
                                                                    truncation=True,
                                                                    return_tensors="pt")
        # generate seqence using beam search sample 
        sampled_outputs = self.t5.generate_for_rl_training(input_ids=inputs_encoding_for_generate.input_ids.cuda(), 
                                                        do_sample=True, 
                                                        top_k=20, 
                                                        top_p=0.95, 
                                                        max_length=self.hparams.max_oupt_len, 
                                                        num_return_sequences=1, 
                                                        num_beams=1,
                                                        output_scores=True,
                                                        return_dict_in_generate=True)
        sampled_seqs = sampled_outputs.sequences
        scores = sampled_outputs.scores

        log_probs = []
        for batch_id in range(sampled_seqs.size(0)):
            seq = sampled_seqs[batch_id]
            log_prob = []
            for i, tok in enumerate(seq):
                if tok not in [0, 1]:
                    score_in_step = scores[i-1]
                    probs = F.softmax(score_in_step, dim=1)
                    log_prob.append(torch.log(probs[batch_id][tok] + 1e-5))
            if len(log_prob) != 0:
                log_prob = torch.stack(log_prob).sum() / len(log_prob)
                log_probs.append(log_prob)
            else:
                log_prob.append(torch.tensor(-100).cuda())
        log_probs = torch.stack(log_probs)
        
        sampled_seqs = self.tokenizer_for_generate.batch_decode(sampled_seqs, skip_special_tokens=True)

        # generate sequence using greedy search
        with torch.no_grad():
            geedy_outputs = self.t5.generate(input_ids=inputs_encoding_for_generate.input_ids.cuda(), 
                                            do_sample=False, 
                                            top_k=20, 
                                            top_p=0.95, 
                                            max_length=self.hparams.max_oupt_len, 
                                            num_return_sequences=1, 
                                            num_beams=1,)
            greedy_seqs = self.tokenizer_for_generate.batch_decode(geedy_outputs, skip_special_tokens=True)

        # reconstruct
        # task_prefix = 'Generate question and context'
        # reconstructed_seqs = self._generate(sampled_seqs, task_prefix)
        # baseline_reconstructed_seqs = self._generate(greedy_seqs, task_prefix, do_sample=False)

        # compute reward
        sample_reward = self.compute_reward(sampled_seqs, gold_output_sentences, gold_inputs_sentences)
        baseline_reward = self.compute_reward(greedy_seqs, gold_output_sentences, gold_inputs_sentences)

        # compute policy loss 
        rl_loss = -(sample_reward - baseline_reward) * log_probs
        rl_loss = torch.mean(rl_loss)
        batch_reward = torch.mean(sample_reward)
        return rl_loss, batch_reward
    
    def compute_reward(self, generated_outputs, gold_outputs, origin_inputs):
        f1_reward = 0
        output_sim_reward = 0
        reconstruct_reward = 0

        #-------------------F1_REWARD-----------------------
        f1_reward = compute_f1(generated_outputs, gold_outputs)[0]
        f1_reward = [f1_reward]*len(generated_outputs)
        f1_reward = torch.tensor(f1_reward, dtype=torch.float)
        #---------------OUTPUT_SIMILAR_REWARD---------------
        output_sim_reward = compute_sentences_similar(generated_outputs, gold_outputs, metric='rouge')
        output_sim_reward = torch.tensor(output_sim_reward, dtype=torch.float)
        #----------------RECONSTRUCT_REWARD-----------------
        with torch.no_grad():
            origin_inputs = self.tokenizer([f"{sent}" for sent in origin_inputs],
                                            padding='longest',
                                            max_length=self.hparams.max_input_len,
                                            truncation=True,
                                            return_tensors="pt").input_ids
            origin_inputs_presentation = self.t5.encoder(input_ids=origin_inputs.cuda()).last_hidden_state[:,0]

            generated_outputs = self.tokenizer(generated_outputs,
                                                padding='longest',
                                                max_length=self.hparams.max_input_len,
                                                truncation=True,
                                                return_tensors="pt").input_ids
            generated_outputs_presentation = self.t5.encoder(input_ids=generated_outputs.cuda()).last_hidden_state[:, 0]

            cosine_scores = util.pytorch_cos_sim(origin_inputs_presentation, generated_outputs_presentation)
            reconstruct_reward = []
            for i in range(len(gold_outputs)):
                reconstruct_reward.append(abs(float(cosine_scores[i][i])))
            
            reconstruct_reward = torch.tensor(reconstruct_reward, dtype=torch.float)
        
        reward = self.hparams.f1_reward_weight * f1_reward  \
                + self.hparams.reconstruct_reward_weight * reconstruct_reward \
                + (1.0 - self.hparams.f1_reward_weight - self.hparams.reconstruct_reward_weight) * output_sim_reward
        return reward.cuda()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        mle_loss = 0
        rl_loss = 0
        batch_reward = 0

        if self.hparams.mle_train:
            mle_loss = self.train_MLE(batch)
        if self.hparams.rl_train:
            rl_loss, batch_reward = self.train_RL(batch)
        
        loss = (1.0 - self.hparams.mle_weight) * rl_loss + self.hparams.mle_weight * mle_loss # need add weights
        self.log_dict({'mle_loss': mle_loss, 'rl_loss': rl_loss, 'reward': batch_reward}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        inputs_sentences = [self.input_formater.format_input(example=example)[-1] for example in batch]
        task_prefix = 'causality identification'

        generated_outputs = self._generate(inputs=inputs_sentences, task_prefix=task_prefix, num_beams=8)
        
        gold_output_sentences = [self.oupt_formater.format_output(example=example) for example in batch]

        return generated_outputs, gold_output_sentences, [f"{task_prefix}\n{sent}" for sent in inputs_sentences]
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
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
        with open('./dev.json','w') as writer:
            writer.write(json.dumps(preds, indent=6)+'\n')
        self.log_dict({'f1_dev': f1, 'p_dev': p, 'r_dev': r}, prog_bar=True)
        return f1

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        inputs_sentences = [self.input_formater.format_input(example=example)[-1] for example in batch]
        task_prefix = 'causality identification'

        generated_outputs = self._generate(inputs=inputs_sentences, task_prefix=task_prefix, num_beams=8)
        
        gold_output_sentences = [self.oupt_formater.format_output(example=example) for example in batch]

        return generated_outputs, gold_output_sentences, [f"{task_prefix}\n{sent}" for sent in inputs_sentences]

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        preds = []
        for output in outputs:
            for sample in zip(*output):
                preds.append({
                    'sentence': sample[2],
                    'predicted': sample[0],
                    'gold': sample[1]
                })

        with open('./test.json','w') as writer:
            writer.write(json.dumps(preds, indent=6)+'\n')

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

