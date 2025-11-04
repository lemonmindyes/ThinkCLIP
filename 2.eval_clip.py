from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoTokenizer

from clip import ThinkCLIP
from config import Config


if __name__ == '__main__':
    config = Config()
    model = ThinkCLIP(config)
    checkpoint = torch.load('./checkpoint.bin')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(config.img_size, interpolation = transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    img = Image.open('girl.jpg').convert('RGB')
    img = transform(img).unsqueeze(0)
    tokenizer = AutoTokenizer.from_pretrained('jingyaogong/MiniMind2')
    eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])
    # text = [
    #     'Happy New Year' + '<|im_end|>',
    #     'China' + '<|im_end|>',
    #     'red' + '<|im_end|>',
    #     'envelope' + '<|im_end|>',
    #     'red envelope' + '<|im_end|>'
    # ]
    # text = [
    #     'pants' + '<|im_end|>',
    #     'plane' + '<|im_end|>',
    #     'car' + '<|im_end|>',
    #     'blue shirt' + '<|im_end|>',
    #     'shoe' + '<|im_end|>',
    #     'Light blue pants' + '<|im_end|>',
    #     'Dark blue pants' + '<|im_end|>',
    # ]
    text = [
        'girl playing mahjong' + '<|im_end|>',
        'boy playing mahjong' + '<|im_end|>',
        'girl' + '<|im_end|>',
        'mahjong' + '<|im_end|>',
        'green mahjong' + '<|im_end|>',
        'playing mahjong' + '<|im_end|>',
    ]
    text_batch = tokenizer.batch_encode_plus(
        text,
        max_length = Config().max_seq_len,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )
    input_ids = text_batch['input_ids'].to(torch.long)
    attention_mask = text_batch['attention_mask'].to(torch.float32)

    logit_img, logit_text = model(img, input_ids, attention_mask, eos_token_id)
    print(logit_img)
    print(logit_text)
    print(torch.softmax(logit_img, dim = -1))