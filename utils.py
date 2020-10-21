from transformers import AutoTokenizer, BertTokenizer
import torch
from onnxruntime.transformers.gpt2_helper import Gpt2Helper
import numpy as np
import random


def gen_sentence(num_exam=10):
    nouns = ("puppy", "car", "rabbit", "girl", "monkey")
    verbs = ("runs", "hits", "jumps", "drives", "barfs")
    adv = ("crazily.", "dutifully.", "foolishly.", "merrily.", "occasionally.")
    adj = ("adorable", "clueless", "dirty", "odd", "stupid")

    results = []
    for i in range(num_exam):
        num = random.randrange(0, 5)
        results.append(nouns[num] + ' ' + verbs[num] + ' ' + adv[num] + ' ' + adj[num])

    return results


def get_tokenizer(model_name_or_path, cache_dir):
    if model_name_or_path == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    if model_name_or_path == "gpt2":
        tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def get_example_inputs(prompt_text=[],
                       model_name_or_path="gpt2",
                       cache_dir="./cache_models/",
                       num_attention_heads=12,
                       hidden_size=768,
                       num_layer=12,
                       device=torch.device("cpu")):
    tokenizer = get_tokenizer(model_name_or_path, cache_dir)
    if model_name_or_path == "gpt2":
        encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)
    else:
        encodings_dict = tokenizer.batch_encode_plus(prompt_text, max_length=128, pad_to_max_length = True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.int64)
    # print(encodings_dict)
    segment_ids = torch.tensor(encodings_dict['token_type_ids'], dtype=torch.int64)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))
    if model_name_or_path == "gpt2":
        return input_ids, attention_mask, position_ids, empty_past
    return input_ids, attention_mask, segment_ids, empty_past


def inference_with_io_binding(session, config, input_ids, position_ids, attention_mask, past, device=torch.device("cpu")):
    output_shapes = Gpt2Helper.get_output_shapes(batch_size=input_ids.size(0),
                                                 past_sequence_length=past[0].size(3),
                                                 sequence_length=input_ids.size(1),
                                                 config=config)
    output_buffers = Gpt2Helper.get_output_buffers(output_shapes, device)

    io_binding = Gpt2Helper.prepare_io_binding(session, input_ids, position_ids, attention_mask, past,
                                               output_buffers, output_shapes)
    session.run_with_iobinding(io_binding)

    outputs = Gpt2Helper.get_outputs_from_io_binding_buffer(session, output_buffers, output_shapes,
                                                            return_numpy=False)
    return outputs


def test_generation(torch_model,
                    tokenizer,
                    config,
                    input_text,
                    ort_session=None,
                    num_tokens_to_produce=30,
                    num_layer=12,
                    device=torch.device("cpu")):
    use_onnxruntime = (ort_session is not None)
    print("Text generation using", "OnnxRuntime" if use_onnxruntime else "PyTorch", "...")
    eos_token_id = tokenizer.eos_token_id

    input_ids, attention_mask, position_ids, past = get_example_inputs(input_text)
    batch_size = input_ids.size(0)

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    for step in range(num_tokens_to_produce):
        if ort_session is not None:
            outputs = inference_with_io_binding(ort_session, config, input_ids, position_ids, attention_mask, past)
        else:
            outputs = torch_model(input_ids, attention_mask=attention_mask, position_ids=position_ids, past=past)

        next_token_logits = outputs[0][:, -1, :]
        # Greedy approach is used here. You can easily extend it to use beam search and sampling to pick next tokens.
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)

        past = []
        if not use_onnxruntime:
            past = list(outputs[1])  # past in torch output is tuple
        else:
            for i in range(num_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if isinstance(outputs[i + 1], np.ndarray) else outputs[
                    i + 1].clone().detach()
                past.append(past_i.to(device))

        if torch.all(has_eos):
            break

    for i, output in enumerate(all_token_ids):
        print("------------")
        print(tokenizer.decode(output, skip_special_tokens=True))