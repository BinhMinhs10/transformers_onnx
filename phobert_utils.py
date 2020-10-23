import torch
from transformers import PhobertTokenizer


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_tokenizer(model_name_or_path, cache_dir):
    tokenizer = PhobertTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def get_example_inputs(prompt_text=[],
                       model_name_or_path=None,
                       cache_dir="./cache_models/"):
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    encodings_dict = tokenizer.batch_encode_plus(prompt_text,
                                                 max_length=128,
                                                 pad_to_max_length=True,
                                                 truncation=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.int64)
    # print(encodings_dict)
    segment_ids = torch.tensor(encodings_dict['token_type_ids'], dtype=torch.int64)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    return input_ids, attention_mask, segment_ids
