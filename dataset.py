import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class GPT2Dataset(Dataset):
    def __init__(self, propmt_list, answer_list, tokenizer,
                max_length_propmt=1024, max_length_answer=10):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.answer_ids = []

        # 设置填充参数为右填充
        tokenizer.padding_side = "right"

        for txt in propmt_list:
            encodings_dict = tokenizer('<s>'+ txt, truncation=True, 
                                     max_length=max_length_propmt, 
                                     padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        
        for ans in answer_list:
            encodings_dict = tokenizer('<s>'+ str(ans), truncation=True,
                                     max_length=max_length_answer,
                                     padding="max_length")
            self.answer_ids.append(torch.tensor(encodings_dict['input_ids']))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.answer_ids[idx]