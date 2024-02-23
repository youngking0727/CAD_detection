import torch
import torch.nn as nn
import json
import multiprocessing
import transformers
from transformers import LlamaTokenizer, EsmModel, EsmConfig, GPT2Tokenizer, GPT2LMHeadModel
#from peft import get_peft_model, LoraConfig, TaskType
from models import ECGClassifier
from models.biomedgpt.base import BioMedGPTBase
from models.biomedgpt.modeling_llama import LlamaForCausalLM


# vanllina biomedgpt
class BioMedGPTV(BioMedGPTBase):
    def __init__(
        self,
        ecg_encoder_ckpt=None,
        freeze_ecg_structure=True,
        llama_ckpt=None,
        llama_peft=False,
        device=None,
    ):
        super(BioMedGPTV, self).__init__()
        self.device = device
        # load mol_structure encoder
        self.ecg_encoder = ECGClassifier(num_classes=5)
        """
        if ecg_encoder_ckpt is not None:
            self.ecg_encoder.load_state_dict(torch.load(
                                                        ecg_encoder_ckpt,
                                                        map_location="cpu"),
                                                        strict=True
                                                        )
        """                                       
        """
        if freeze_ecg_structure:
            print("freeze molecule structure encoder")
            for name, param in self.ecg_encoder.named_parameters():
                param.requires_grad = False
        """
        self.ecg_encoder.to(self.device)
        self.ecg_encoder.encoder.to(self.device)

        # load llm
        self.llm_tokenizer = transformers.AutoTokenizer.from_pretrained(
                                                                    llama_ckpt,
                                                                    # padding_side="right",
                                                                    use_fast=False,
                                                                )
        self.init_prepare_token()
        
        self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm = transformers.AutoModelForCausalLM.from_pretrained(
                                                                      llama_ckpt,
                                                                      #use_cache=True,
                                                                      #torch_dtype=torch.bfloat16,
                                                                      #use_flash_attention_2=True
                                                                    )
        #self.llm.to(self.device)                                           
        #self.llm.resize_token_embeddings(len(self.llm_tokenizer))
        if llama_peft:
            lora_config = LoraConfig(peft_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05, target_modules=["v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"])
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
            for name, param in self.llm.named_parameters():
                if param.requires_grad:
                    print(name)
        
        print("freeze llm")
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
        
        self.proj_ecg = nn.Linear(512, self.llm.config.hidden_size)


        self.classify_prompt = """
        Based on the given ECG signal above, please identify which type (A-E) it belongs to.

        A. Normal
        B. Myocardial Infarction
        C. ST/T Change
        D. Conduction Disturbance
        E. Hypertrophy
        
        Please identify and choose one of A or B or C or D or E to answer strictly
        """

        self.rhythm_prompt = """
        Based on the given ECG signal above, please identify which rhythm type it belongs to.

        A: sinus rhythm or sinus irregularity
        B: Atrial Fibrillation
        C: supraventricular tachycardia, atrial tachycardia, atrioventricular node reentrant tachycardia, atrioventricular reentrant tachycardia or sinus atrium to atrial wandering rhythm
        D: sinus bradycardia
        
        Please identify and choose one of AFIB, GSVT, SR, SB to answer strictly
        """

        self.ptb_rhythm_prompt = """
        Based on the given ECG signal above, please identify which rhythm type it belongs to.

        A. Sinus Rhythm
        B. atrial fibrillation
        C. sinus tachycardia
        D. sinus arrhythmia
        E. sinus bradycardia
        
        Please identify and choose one of A or B or C or D or E to answer strictly
        """

        self.form_prompt = """
        Based on the given ECG signal above, please identify which form type it belongs to.

        A. non-diagnostic T abnormalities
        B. non-specific ST changes
        C. digitalis-effect
        D. long QT-interval
        E. abnormal QRS
        
        Please identify and choose one of A or B or C or D or E to answer strictly
        """

    def init_prepare_token(self):
        self.prepare_token = {
                    "lead V1: ": self.llm_tokenizer("lead V1: ",return_tensors='pt', add_special_tokens=False),
                    "lead V2: ": self.llm_tokenizer("lead V2: ",return_tensors='pt', add_special_tokens=False),
                    "lead V3: ": self.llm_tokenizer("lead V3: ",return_tensors='pt', add_special_tokens=False),
                    "lead V4: ": self.llm_tokenizer("lead V4: ",return_tensors='pt', add_special_tokens=False),
                    "lead V5: ": self.llm_tokenizer("lead V5: ",return_tensors='pt', add_special_tokens=False),
                    "lead V6: ": self.llm_tokenizer("lead V5: ",return_tensors='pt', add_special_tokens=False),
                    "lead I: ": self.llm_tokenizer("lead I: ",return_tensors='pt', add_special_tokens=False),
                    "lead II: ": self.llm_tokenizer("lead II: ",return_tensors='pt', add_special_tokens=False),
                    "lead III: ": self.llm_tokenizer("lead III: ",return_tensors='pt', add_special_tokens=False),
                    "lead aVL: ": self.llm_tokenizer("lead aVL: ",return_tensors='pt', add_special_tokens=False),
                    "lead aVR: ": self.llm_tokenizer("lead aVR: ",return_tensors='pt', add_special_tokens=False),
                    "lead aVF: ": self.llm_tokenizer("lead aVF: ",return_tensors='pt', add_special_tokens=False),
                    " ": self.llm_tokenizer(" ",return_tensors='pt', add_special_tokens=False),
                    "are given. ": self.llm_tokenizer("are given. ",return_tensors='pt', add_special_tokens=False),
                }

    def add_padding(self, wrapped_embeds, wrapped_attention_mask, targets=None, padding="right"):
        batch_size = len(wrapped_embeds)
        max_length_batch = 0
        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] > max_length_batch:
                max_length_batch = wrapped_embeds[i].shape[1]

        for i in range(batch_size):
            if wrapped_embeds[i].shape[1] < max_length_batch:
                pad_len = max_length_batch - wrapped_embeds[i].shape[1]
                pad_token = torch.ones((1, 1), dtype=torch.long, device=wrapped_embeds[i].device) * self.llm_tokenizer.pad_token_id
                pad_token_embeds = self.llm.get_input_embeddings()(pad_token)
                # pad_token_embeds = torch.zeros((1, 1, wrapped_embeds[i].shape[2]), dtype=wrapped_embeds[i].dtype, device=wrapped_embeds[i].device)
                if padding == "right":
                    wrapped_embeds[i] = torch.cat((
                        wrapped_embeds[i], 
                        pad_token_embeds.expand(-1, pad_len, -1)
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
                        pad_token_embeds.expand(-1, pad_len, -1),
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
        wrapped_embeds, wrapped_attention_mask, wrapped_targets = [], [], []
        for data in samples:
            start = "<s>" + data['prompt1'] + data['prompt2']
            end = data['prompt3'] + ". " + data['prompt4']
            start_token = self.llm_tokenizer(start, return_tensors='pt', add_special_tokens=False).to(self.device)
            # input_ids.ne(tokenizer.pad_token_id)
            start_embeds = self.llm.get_input_embeddings()(start_token.input_ids)
            end_token = self.llm_tokenizer(end,return_tensors='pt',add_special_tokens=False).to(self.device)
            # input_ids.ne(tokenizer.pad_token_id)
            end_embeds = self.llm.get_input_embeddings()(end_token.input_ids)
            for signal in data['encoding']:
                lead_token = self.prepare_token[f"lead {signal}: "].to(self.device)
                start += f"lead {signal}: "
                lead_embeds = self.llm.get_input_embeddings()(lead_token.input_ids)
                start_embeds = torch.cat([start_embeds, lead_embeds], dim=1)
                lead_values = torch.tensor(data['signal_data'][signal]['value'].reshape([1, 1, 5000])).float().to(self.device)
                lead_values = self.ecg_encoder.get_feature(lead_values).unsqueeze(1)
                start += f"<signal> "
                # 更改维度
                lead_values = self.proj_ecg(lead_values)
                start_embeds = torch.cat([start_embeds, lead_values], dim=1)
                description = data['signal_data'][signal]['description'] if 'description' in data['signal_data'][signal].keys() else ""
                if description:
                    end += description + ", "
            start += "are given. "
            s_end_token = self.prepare_token["are given. "].to(self.device)
            s_end_embeds = self.llm.get_input_embeddings()(s_end_token.input_ids)
            start_embeds = torch.cat([start_embeds, s_end_embeds], dim=1)
            end = end[:-2] + ".</s>"
            end_token = self.llm_tokenizer(end,return_tensors='pt',add_special_tokens=False).to(self.device)
            end_embeds = self.llm.get_input_embeddings()(end_token.input_ids)
            embeds = torch.cat([start_embeds, end_embeds], dim=1)
            # mask
            # mask只需要维度一致就好，都是1就行
            masks = torch.ones(embeds.shape[:-1]).to(self.device)
            # label
            # instruction部分都是-100,后半部分正常
            labels_start = torch.ones(start_embeds.shape[:-1], dtype=torch.long).to(self.device).fill_(-100)
            labels_end = end_token.input_ids.masked_fill(end_token.input_ids == self.llm_tokenizer.pad_token_id, -100)
            labels = torch.cat([labels_start, labels_end], dim=1).to(self.device)
            wrapped_embeds.append(embeds)
            wrapped_attention_mask.append(masks)
            wrapped_targets.append(labels)

        inputs_embeds, inputs_attention_mask, targets = self.add_padding(wrapped_embeds, wrapped_attention_mask, wrapped_targets)

        #with torch.autocast("cuda"):
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs_attention_mask,
            labels=targets,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=True
        )

        return outputs.loss
       

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=512,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=0.01,
        task="classify"
    ):

        # TODO:
        single_lead = False
        wrapped_embeds, wrapped_attention_mask = [], []
        for data in samples:
            if "prompt1" in data.keys():
                start = "<s>" + data['prompt1'] + data['prompt2']
            else:
                start = "<s>" + "The ecg signal of "
            if 'prompt3' in data.keys():
                label = data['prompt3'] + ". " + data['prompt4']
            else:
                end = "The diagnosis of this ECG may be one of the following: A: normal ECG B: Myocardial Infarction C: ST/T Change D: Conduction Disturbance E: Hypertrophy. Please just answer using A, B, C, D, or E. The answer is:"
            # TODO:
            if task == "classify":
                end = self.classify_prompt
            elif task == "ptb_rhythm":
                end = self.ptb_rhythm_prompt
            elif task == "form":
                end = self.form_prompt
            elif task == "rhythm":
                end = self.rhythm_prompt
            # end = " "
            start_token = self.llm_tokenizer(start, return_tensors='pt', add_special_tokens=False).to(self.device)
            # input_ids.ne(tokenizer.pad_token_id)
            start_embeds = self.llm.get_input_embeddings()(start_token.input_ids)
            end_token = self.llm_tokenizer(end,return_tensors='pt',add_special_tokens=False).to(self.device)
            # input_ids.ne(tokenizer.pad_token_id)
            end_embeds = self.llm.get_input_embeddings()(end_token.input_ids)
            if not single_lead:
                choose_list = ['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']
            else:
                choose_list = ["II"]
            for signal in choose_list:
                lead_token = self.prepare_token[f"lead {signal}: "].to(self.device)
                start += f"lead {signal}: "
                lead_embeds = self.llm.get_input_embeddings()(lead_token.input_ids)
                start_embeds = torch.cat([start_embeds, lead_embeds], dim=1)
                lead_values = torch.tensor(data['signal_data'][signal]['value'].reshape([1, 1, 5000])).float().to(self.device)
                lead_values = self.ecg_encoder.get_feature(lead_values).unsqueeze(1)
                start += f"<signal> "
                # 更改维度
                lead_values = self.proj_ecg(lead_values)
                start_embeds = torch.cat([start_embeds, lead_values], dim=1)
                description = data['signal_data'][signal]['description'] if 'description' in data['signal_data'][signal].keys() else ""
                if description:
                    label += description + ", "
            start += "are given. "
            label = label[:-2] + ".</s>"
            print("label: ", label, data["signal_data"]['diagnostic_class'])
            s_end_token = self.prepare_token["are given. "].to(self.device)
            s_end_embeds = self.llm.get_input_embeddings()(s_end_token.input_ids)
            start_embeds = torch.cat([start_embeds, s_end_embeds], dim=1)


            embeds = torch.cat([start_embeds, end_embeds], dim=1)
            # mask
            # mask只需要维度一致就好，都是1就行
            masks = torch.ones(embeds.shape[:-1]).to(self.device)

            wrapped_embeds.append(embeds)
            wrapped_attention_mask.append(masks)

        inputs_embeds, attention_mask = self.add_padding(wrapped_embeds, wrapped_attention_mask, padding="left")
        
        with torch.autocast("cuda"):
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=0.1,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text