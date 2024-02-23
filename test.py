import pickle
import random
import torch
import transformers
from models import ECGClassifier
from torch.utils.data import Dataset, DataLoader


class MultECGDataset(Dataset):
    def __init__(self, signal_data, channel_data, index):
        self.signal_data = signal_data
        self.channel_data = channel_data
        self.index = []
        for i in index:
            for channel_data in self.channel_data[i]['encoding']:
                self.index.append([i, channel_data])
        random.shuffle(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        person_id_channel = self.index[index]
        person_id = person_id_channel[0]
        channel_data = person_id_channel[1]
        return {
            "person_id": person_id,
            "prompt1": self.channel_data[person_id]['prompt1'],
            "prompt2": self.channel_data[person_id]['prompt2'],
            "prompt3": self.channel_data[person_id]['prompt3'],
            "prompt4": self.channel_data[person_id]['prompt4'],
            "encoding": channel_data,
            "signal_data": signal_data[person_id]
        }


class DataCollator(object):

    def __call__(self, instances):
        return instances


if __name__ == "__main__":

    channel_data_path = "/AIRvePFS/dair/users/hongcd/home/ECG_Data/processed/permutation_dico.pkl"
    signal_data_path = "/AIRvePFS/dair/users/hongcd/home/ECG_Data/processed/info_dict.pkl"

    with open(signal_data_path, 'rb') as file:
        signal_data = pickle.load(file)
    with open(channel_data_path, 'rb') as file:
        channel_data = pickle.load(file)

    keys = list(signal_data.keys())
    random.shuffle(keys)
    split = int(len(keys) * 0.95)
    train_indices = keys[:split]
    val_indices = keys[split:]

    dataset = MultECGDataset(signal_data, channel_data, train_indices)
    print(dataset[0])

    collator = DataCollator()
    loader = DataLoader(
        dataset, 
        batch_size=8, 
        collate_fn=collator
        )

    for batch_data in loader:
        print(batch_data)
        break

    llama_ckpt = "/AIRvePFS/dair/ckpts/BioMedGPT-LM-7B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
                                                        llama_ckpt,
                                                        model_max_length=2048,
                                                        padding_side="right",
                                                        use_fast=False,
                                                        )
    tokenizer.pad_token = tokenizer.unk_token

    llm = transformers.AutoModelForCausalLM.from_pretrained(
                                                        llama_ckpt,
                                                        torch_dtype=torch.bfloat16,
                                                        #use_flash_attention_2=True
                                                        )

    ecg_encoder = ECGClassifier(num_classes=5)
    ecg_encoder_ckpt = "./ckpts/0128.pth"
    ecg_encoder.load_state_dict(torch.load(ecg_encoder_ckpt, map_location="cpu"), strict=True)

    prepare_token = {
                    "lead V1: ": tokenizer("lead V1: ",return_tensors='pt', add_special_tokens=False),
                    "lead V2: ": tokenizer("lead V2: ",return_tensors='pt', add_special_tokens=False),
                    "lead V3: ": tokenizer("lead V3: ",return_tensors='pt', add_special_tokens=False),
                    "lead V4: ": tokenizer("lead V4: ",return_tensors='pt', add_special_tokens=False),
                    "lead V5: ": tokenizer("lead V5: ",return_tensors='pt', add_special_tokens=False),
                    "lead V6: ": tokenizer("lead V5: ",return_tensors='pt', add_special_tokens=False),
                    "lead I: ": tokenizer("lead I: ",return_tensors='pt', add_special_tokens=False),
                    "lead II: ": tokenizer("lead II: ",return_tensors='pt', add_special_tokens=False),
                    "lead III: ": tokenizer("lead III: ",return_tensors='pt', add_special_tokens=False),
                    "lead aVL: ": tokenizer("lead aVL: ",return_tensors='pt', add_special_tokens=False),
                    "lead aVR: ": tokenizer("lead aVR: ",return_tensors='pt', add_special_tokens=False),
                    "lead aVF: ": tokenizer("lead aVF: ",return_tensors='pt', add_special_tokens=False),
                    " ": tokenizer(" ",return_tensors='pt', add_special_tokens=False),
                    "are given. ": tokenizer("are given. ",return_tensors='pt', add_special_tokens=False),
                }

    device = llm.device
    linear_projection = torch.nn.Linear(512, 4096)

    wrapped_embeds, wrapped_attention_mask, targets = [], [], []


    for data in batch_data:
        start = "<s>" + data['prompt1'] + data['prompt2']
        end = data['prompt3'] + ". " + data['prompt4']
        start_token = tokenizer(start,return_tensors='pt',add_special_tokens=False).to(device)
        # input_ids.ne(tokenizer.pad_token_id)
        start_embeds = llm.get_input_embeddings()(start_token.input_ids).to(device)
        end_token = tokenizer(end,return_tensors='pt',add_special_tokens=False).to(device)
        # input_ids.ne(tokenizer.pad_token_id)
        end_embeds = llm.get_input_embeddings()(end_token.input_ids).to(device)
        for signal in data['encoding']:
            lead_token = prepare_token[f"lead {signal}: "]
            start += f"lead {signal}: "
            lead_embeds = llm.get_input_embeddings()(lead_token.input_ids).to(device)
            start_embeds = torch.cat([start_embeds, lead_embeds], dim=1).to(device)
            lead_values = torch.tensor(data['signal_data'][signal]['value'].reshape([1, 1, 5000])).float()
            lead_values = ecg_encoder.get_feature(lead_values).unsqueeze(1)
            start += f"<signal> "
            # 更改维度
            lead_values = linear_projection(lead_values)
            start_embeds = torch.cat([start_embeds, lead_values], dim=1).to(device)
            description = data['signal_data'][signal]['description'] if 'description' in data['signal_data'][signal].keys() else ""
            if description:
                end += description + ", "
        start += "are given. "
        s_end_token = prepare_token["are given. "]
        s_end_embeds = llm.get_input_embeddings()(s_end_token.input_ids).to(device)
        start_embeds = torch.cat([start_embeds, s_end_embeds], dim=1).to(device)
        end = end[:-2] + ".</s>"
        end_token = tokenizer(end,return_tensors='pt',add_special_tokens=False).to(device)
        end_embeds = llm.get_input_embeddings()(end_token.input_ids).to(device)
        embeds = torch.cat([start_embeds, end_embeds], dim=1).to(device)
        # mask
        # mask只需要维度一致就好，都是1就行
        masks = torch.ones(embeds.shape[:-1]).to(device)
        # label
        # instruction部分都是-100,后半部分正常
        labels_start = torch.ones(start_embeds.shape[:-1], dtype=torch.long).to(device).fill_(-100)
        labels_end = end_token.input_ids.masked_fill(end_token.input_ids == tokenizer.pad_token_id, -100)
        labels = torch.cat([labels_start, labels_end], dim=1).to(device)
        wrapped_embeds.append(embeds)
        wrapped_attention_mask.append(masks)
        targets.append(labels)

    batch_size = len(targets)
    max_length_batch = 0
    for i in range(batch_size):
        if wrapped_embeds[i].shape[1] > max_length_batch:
            max_length_batch = wrapped_embeds[i].shape[1]
    
    padding = "right"
    for i in range(batch_size):
        if wrapped_embeds[i].shape[1] < max_length_batch:
            pad_len = max_length_batch - wrapped_embeds[i].shape[1]
            pad_token = torch.ones((1, 1), dtype=torch.long, device=wrapped_embeds[i].device) * tokenizer.pad_token_id
            pad_token_embeds = llm.get_input_embeddings()(pad_token)
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
        a = torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0), torch.cat(targets, dim=0)
    else:
        a = torch.cat(wrapped_embeds, dim=0), torch.cat(wrapped_attention_mask, dim=0)
            
    print("end")