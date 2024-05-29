from transformers import DataCollatorForLanguageModeling, EsmForMaskedLM, EsmTokenizer, EsmConfig
from data.make_data import make_esm_dataset, DataCollatorForFIM
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import wandb
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import re


def find_latest_checkpoint(path_pattern):
    files = [f for f in os.listdir('.') if re.match(path_pattern, f)]
    if not files:
        return None, 0

    numbers = [int(re.search(r'\d+', f).group()) for f in files]
    max_num = max(numbers)

    return f"./esm_FIM_{max_num}.pth", max_num


def setup_wandb():
    wandb.login(key='353758ac65c9ac5ceab0c5b51ce078ea9176161d')
    wandb.init(project='diploma', entity='stasstaf')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def train(rank, world_size):
    mode = 'FIM'
    use_ddp = world_size != 0
    use_checkpoint = False
    testing = True
    use_wandb = False
    if use_ddp:
        setup(rank, world_size)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    if testing:
        tokenizer = EsmTokenizer(r'..\data\raw\vocab.txt')
    else:
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm-1b")
    if mode == 'FIM':
        data_collator = DataCollatorForFIM(
            tokenizer=tokenizer,
            mlm=True
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )
    file_path = r"..\data\raw\AFDBv4_90.128-254.fasta"
    train_dataset, val_dataset, _ = make_esm_dataset(file_path, tokenizer)

    if testing:
        cfg = EsmConfig(vocab_size=len(tokenizer), mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id)
        batch_size = 1
        cfg.max_position_embeddings = 256
        cfg.intermediate_size = 128
        cfg.hidden_size = 128
        cfg.num_attention_heads = 1
        cfg.num_hidden_layers = 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)

    else:
        cfg = EsmConfig.from_pretrained("facebook/esm-1b")
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        batch_size = 256
        cfg.max_position_embeddings = 256
        cfg.intermediate_size = 768
        cfg.hidden_size = 768
        cfg.num_attention_heads = 12
        cfg.num_hidden_layers = 12
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=data_collator)

    model = EsmForMaskedLM(cfg).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    checkpoint_pattern = r'esm_FIM_\d+\.pth'
    checkpoint_path, max_number = find_latest_checkpoint(checkpoint_pattern)
    start_epoch = 0
    if use_checkpoint and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = max_number + 1
    if use_wandb and (not use_ddp or is_main_process()):
        setup_wandb()
    model.train()

    model.train()
    step = 0
    max_epoch = 1025
    save_freq = 64 if not testing else 1000
    for epoch in range(start_epoch, max_epoch):
        if use_ddp:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs = batch['input_ids']
            labels = batch['labels']
            if use_ddp:
                inputs = inputs.to(rank)
                labels = labels.to(rank)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0 and (not use_ddp or is_main_process()):
                model.eval()
                val_batch = next(iter(val_loader))
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                with torch.no_grad():
                    val_outputs = model(**val_batch)
                    val_loss = val_outputs.loss
                    metrics = {
                        "Step": step,
                        "Train Loss": loss.item(),
                        "Validation Loss": val_loss.item(),
                        "Epoch": epoch
                    }
                    if use_wandb:
                        wandb.log(metrics)
                    else:
                        print(metrics)
                model.train()
            step += 1
        if (not use_ddp or is_main_process()) and epoch % save_freq == 0:
            path = f"./esm_FIM_{epoch}.pth"
            state_dict = model.module.state_dict() if use_ddp else model.state_dict()
            state = {'model': state_dict,
                     'optimizer': optimizer.state_dict(),
                     }
            torch.save(state, path)
    if use_wandb:
        wandb.finish()
    if use_ddp:
        cleanup()
    print("Training completed for rank: ", rank)


def main():
    world_size = torch.cuda.device_count()
    # assert world_size >= 2, f"Requires at least 2 GPUs to run, but got {world_size}"
    if world_size > 0:
        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(0, 0)


if __name__ == "__main__":
    main()
