import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import Config
from clip import ThinkCLIP
from dataset import LLaVAPretrainDataset

tokenizer = AutoTokenizer.from_pretrained('jingyaogong/MiniMind2')


def collate_fn(batch):
    img = [v[0] for v in batch]
    text = [v[1] + '<|im_end|>' for v in batch]

    text_batch = tokenizer.batch_encode_plus(
        text,
        max_length = Config().max_seq_len,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt',
    )
    img = torch.stack(img)
    input_ids = text_batch['input_ids'].to(torch.long)
    attention_mask = text_batch['attention_mask'].to(torch.float32)
    return img, input_ids, attention_mask


if __name__ == '__main__':
    # 参数
    dtype = torch.bfloat16
    batch_size = 128
    base_lr = 1e-3 * (batch_size / 256)
    epoch = 32
    warmup_epoch = 2

    eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = nullcontext() if device.type == 'cpu' else torch.amp.autocast(device_type = device.type, dtype = dtype)

    config = Config()
    config.vocab_size = tokenizer.vocab_size
    model = ThinkCLIP(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total params:{total_params}')

    train_dataset = LLaVAPretrainDataset(config, "E:/data-hub/LLaVA-Pretrain")
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn,
                              num_workers = 4, pin_memory = False, persistent_workers = True)

    loss_func = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled = (device.type == 'cuda'))
    opt = torch.optim.AdamW(model.parameters(), lr = base_lr, betas = (0.9, 0.98), weight_decay = 0.2)

    total_step = epoch * len(train_loader)
    warmup_step = warmup_epoch * len(train_loader)
    if not os.path.exists('train_log.txt'):
        with open('train_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'Epoch, Step, Loss, Lr, Time\n')
    try:
        checkpoint = torch.load('checkpoint.bin')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['opt'])
        scaler.load_state_dict(checkpoint['scaler'])
    except:
        checkpoint = {'epoch': 0, 'global_step': 0}
    global_step = checkpoint['global_step']

    for e in range(checkpoint['epoch'], epoch):
        model.train()
        start_time = time.time()
        for step, (img, text, attn_mask) in enumerate(train_loader):
            if e < warmup_epoch:
                lr = base_lr * global_step / warmup_step
            else:
                progress = (global_step - warmup_step) / (total_step - warmup_step)
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

            # update lr
            for param_group in opt.param_groups:
                param_group['lr'] = lr

            img, text, attn_mask = img.to(device), text.to(device), attn_mask.to(device)
            with ctx:
                logit_img, logit_text = model(img, text, attn_mask, eos_token_id)
                label = torch.arange(img.shape[0], device = device)
                loss1 = loss_func(logit_img, label)
                loss2 = loss_func(logit_text, label)
                loss = (loss1 + loss2) / 2

            opt.zero_grad(set_to_none = True)  # zero gradient
            scaler.scale(loss).backward()  # scale loss to fit opt's expected gradient scale
            scaler.unscale_(opt)  # unscale gradient
            scaler.step(opt)  # step opt
            scaler.update()  # update scaler

            global_step += 1

            if step % 20 == 0:
                print(f'Epoch:{e + 1}/{epoch}, Step:{step + 1}/{len(train_loader)}, Loss:{loss.item():.4f}, '
                      f'Lr:{opt.param_groups[0]["lr"]:.6f}, Time:{time.time() - start_time:.2f}')
                with open('train_log.txt', 'a', encoding = 'utf-8') as f:
                    f.write(f'{e + 1}, {step + 1}, {loss.item():.4f}, {opt.param_groups[0]["lr"]:.6f}, '
                            f'{time.time() - start_time:.2f}\n')

        checkpoint = {
            'epoch': e,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'global_step': global_step
        }
        torch.save(checkpoint, 'checkpoint.bin.pt')
        os.replace('checkpoint.bin.pt', 'checkpoint.bin')



