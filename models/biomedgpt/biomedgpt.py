import json
import re

import torch
import torch.nn as nn

#from peft import get_peft_model, LoraConfig, TaskType
from transformers import LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2LMHeadModel
from models.biomedgpt.base import BioMedGPTBase

# biomedgpt
class BioMedGPT(BioMedGPTBase):
    def __init__(
        self,
        mol_qformer_ckpt=None, 
        mol_qformer_config=None, 
        prot_qformer_ckpt=None,
        prot_qformer_config=None,
        llama_ckpt=None,
        llama_config=None,
        device=None
    ):
        super(BioMedGPT, self).__init__()
        self.mol_qformer_config = mol_qformer_config
        self.prot_qformer_config = prot_qformer_config

        self.device = device
        # load molecule qformer
        self.mol_qformer = MolQFormer(
            structure_ckpt=mol_qformer_config["structure_ckpt"],
            structure_config=mol_qformer_config["structure_config"],
            freeze_structure=True,
            qformer_config=mol_qformer_config["qformer_config_file"],
            max_n_nodes=mol_qformer_config["max_n_nodes"],
            tokenizer=mol_qformer_config["tokenizer_name_or_path"],
            max_seq_len=mol_qformer_config["max_seq_len"],
            use_kg=mol_qformer_config["use_kg"],
            device=device,
            mode="finetune",
        ) 
        if mol_qformer_ckpt is not None:
            self.mol_qformer.load_state_dict(torch.load(mol_qformer_ckpt, map_location="cpu")["model"], strict=True)
        # load protein qformer
        self.prot_qformer = ProtQFormer(
            structure_model=prot_qformer_config["protein"]["structure_model"],
            freeze_structure=True,
            lora_structure=prot_qformer_config["protein"]["lora"],
            qformer_config=prot_qformer_config["qformer_config_file"],
            tokenizer=prot_qformer_config["tokenizer_name_or_path"],
            use_kg=prot_qformer_config["use_kg"],
            device=device,
            mode="finetune"
        )
        if prot_qformer_ckpt is not None:
            self.prot_qformer.load_state_dict(torch.load(prot_qformer_ckpt, map_location="cpu")["model"], strict=True)
        # TODO: load cell qformer

        # load llm
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llama_ckpt, use_fast=False, truncation_side="left")
        #self.llm_tokenizer = GPT2Tokenizer.from_pretrained(llama_ckpt, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '<unk>'})
        print("loading llm")
        self.llm = LlamaForCausalLM.from_pretrained(llama_ckpt, torch_dtype=torch.float16)
        #self.llm = GPT2LMHeadModel.from_pretrained(llama_ckpt, torch_dtype=torch.float16)
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))
        lora_config = LoraConfig(peft_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05, target_modules=["v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        print("freeze llm")
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
        self.proj_mol = nn.Linear(self.mol_qformer.qformer_config.hidden_size, self.llm.config.hidden_size)
        self.proj_prot = nn.Linear(self.prot_qformer.qformer_config.hidden_size, self.llm.config.hidden_size)

    def prompt_wrap(self, mol_feats, prot_feats, cell_feats, text_input, prompt):
        device = text_input.input_ids.device

        batch_size = mol_feats.shape[0]
        wrapped_embeds_batch, wrapped_attention_mask_batch = [], []
        cur_mol, cur_prot, cur_cell = 0, 0, 0
        for i in range(batch_size):
            text = prompt[i].format(text_input=text_input[i])
            bos_token = torch.ones((1, 1), dtype=text_input.input_ids.dtype, device=text_input.input_ids.device)
            wrapped_embeds = [bos_token * self.llm_tokenizer.bos_token_id]
            pattern = re.compile("<moleculeHere>|<proteinHere>|<cellHere>")
            p_text = pattern.split(text)
            spec_tokens = pattern.findall(text)
            for j in range(len(p_text)):
                p_tokens = self.llm_tokenizer(
                    p_text[j],
                    return_tensors='pt',
                    add_special_tokens=False
                ).to(device)
                p_embeds = self.llm.get_input_embeddings()(p_tokens.input_ids)
                wrapped_embeds.append(p_embeds)
                if j < len(spec_tokens):
                    if spec_tokens[j] == "<moleculeHere>":
                        wrapped_embeds.append(mol_feats[cur_mol])
                        cur_mol += 1
                    elif spec_tokens[j] == "<proteinHere>":
                        wrapped_embeds.append(prot_feats[cur_prot])
                        cur_prot += 1
                    elif spec_tokens[j] == "<cellHere>":
                        wrapped_embeds.append(cell_feats[cur_cell])
                        cur_cell += 1
            wrapped_embeds_batch.append(wrapped_embeds)
            wrapped_attention_mask_batch.append(torch.ones(wrapped_embeds[-1].shape[:-1]).to(device))
        return wrapped_embeds_batch, wrapped_attention_mask_batch

    def add_padding(self, wrapped_embeds, wrapped_attention_mask, targets=None, padding="right"):
        batch_size = len(wrapped_embeds)
        max_length_batch = 0
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] > max_length_batch:
                max_length_batch = wrapped_embeds[i].shape[1]
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] < max_length_batch:
                pad_len = max_length_batch - wrapped_embeds[i].shape[1]
                if padding == "right":
                    wrapped_embeds[i] = torch.cat((
                        wrapped_embeds[i], 
                        torch.zeros((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype).to(wrapped_embeds[i].device)
                    ), dim=1)
                    wrapped_attention_mask[i] = torch.cat((
                        wrapped_attention_mask[i],
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device)
                    ), dim=1)
                    if targets is not None:
                        targets[i] = torch.cat((
                            targets[i],
                            torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100)
                        ), dim=1)
                else:
                    wrapped_embeds[i] = torch.cat((
                        torch.zeros((1, pad_len, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype).to(wrapped_embeds[i].device),
                        wrapped_embeds[i], 
                    ), dim=1)
                    wrapped_attention_mask[i] = torch.cat((
                        torch.zeros((1, pad_len), dtype=wrapped_attention_mask[i].dtype).to(wrapped_attention_mask[i].device),
                        wrapped_attention_mask[i],
                    ), dim=1)
                    if targets is not None:
                        targets[i] = torch.cat((
                            torch.ones((1, pad_len), dtype=targets[i].dtype).to(targets[i].device).fill_(-100),
                            targets[i],
                        ), dim=1)
        if targets is not None:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0), torch.cat(targets, dim=0)
        else:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0)


    def forward(self, samples):
        with self.maybe_autocast():
            if "mol" in samples:
                text_inputs_mol_qformer = self.mol_qformer.tokenizer(
                    samples["text_inputs"],
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.mol_qformer.max_seq_len
                ).to(samples["mol"].x.device)
                mol_feats = self.mol_qformer(samples["mol"], text_inputs_mol_qformer)
                mol_feats = self.proj_mol(mol_feats)
            else:
                mol_feats = None
            if "protein" in samples:
                text_inputs_prot_qformer = self.prot_qformer.tokenizer(
                    samples["text_inputs"],
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.mol_qformer.max_seq_len
                ).to(samples["protein"].input_ids.device)
                prot_feats = self.prot_qformer(samples["protein"], text_inputs_prot_qformer)
                prot_feats = self.proj_prot(prot_feats)
            else:
                prot_feats = None
            if "cell" in samples:
                pass
            else:
                cell_feats = None

        inputs_embeds, inputs_attention_mask = self.prompt_wrap(mol_feats, prot_feats, cell_feats, samples["text_inputs"], samples["prompt"])
        
        wrapped_embeds, wrapped_attention_mask, wrapped_targets = [], [], []
        for i in range(len(inputs_embeds)):
            output_tokens = self.llm_tokenizer(
                samples["text_outputs"][i],
                return_tensors='pt',
                add_special_tokens=False
            ).to(inputs_embeds[i].device)
            eos_token = torch.ones((1, 1), dtype=output_tokens.input_ids.dtype, device=output_tokens.input_ids.device)
            output_tokens.input_ids = torch.cat([output_tokens.input_ids, eos_token * self.llm_tokenizer.eos_token_id], dim=1)
            output_tokens.attention_mask = torch.cat([output_tokens.attention_mask, eos_token], dim=1)
            output_embeds = self.llm.get_input_embeddings()(output_tokens.input_ids)
            wrapped_embeds.append(torch.cat([inputs_embeds[i], output_embeds], dim=1))
            wrapped_attention_mask.append(torch.cat([inputs_attention_mask[i], output_tokens.attention_mask], dim=1))
            # do not apply loss to the padding
            targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
            )
            # do not apply loss to the text input (i.e., instruction)
            empty_targets = torch.ones(inputs_attention_mask[i].shape, dtype=torch.long).to(inputs_embeds[i].device).fill_(-100)
            wrapped_targets.append(torch.cat([empty_targets, targets], dim=1))
            
        inputs_embeds, inputs_attention_mask, targets = self.add_padding(wrapped_embeds, wrapped_attention_mask, wrapped_targets)
        with self.maybe_autocast():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs_attention_mask,
                labels=targets,
                return_dict=True
            )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            if "mol" in samples:
                text_inputs_mol_qformer = self.mol_qformer.tokenizer(
                    samples["text_inputs"],
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.mol_qformer.max_seq_len
                ).to(samples["mol"].x.device)
                mol_feats = self.mol_qformer(samples["mol"], text_inputs_mol_qformer)
                mol_feats = self.proj_mol(mol_feats)
            else:
                mol_feats = None
            if "protein" in samples:
                text_inputs_prot_qformer = self.prot_qformer.tokenizer(
                    samples["text_inputs"],
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding='max_length',
                    max_length=self.mol_qformer.max_seq_len
                ).to(samples["protein"].input_ids.device)
                prot_feats = self.prot_qformer(samples["protein"], text_inputs_prot_qformer)
                prot_feats = self.proj_prot(prot_feats)
            else:
                prot_feats = None
            if "cell" in samples:
                pass
            else:
                cell_feats = None

            wrapped_embeds, wrapped_attention_mask = self.prompt_wrap(mol_feats, prot_feats, cell_feats, samples["text_inputs"], samples["prompt"])
            inputs_embeds, attention_mask = self.add_padding(wrapped_embeds, wrapped_attention_mask, padding="left")
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
        
        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text