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
    setup(rank, world_size)
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
    file_path = "data/raw/AFDBv4_90.128-254.fasta"
    train_dataset, val_dataset, _ = make_esm_dataset(file_path, tokenizer)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=data_collator)
    device = torch.device("cuda", rank)
    print(device)
    cfg = EsmConfig.from_pretrained("facebook/esm-1b")
    cfg.max_position_embeddings = 256
    cfg.intermediate_size = 768
    cfg.hidden_size = 768
    cfg.num_attention_heads = 12
    cfg.num_hidden_layers = 12
    model = EsmForMaskedLM(cfg).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    if is_main_process():
        setup_wandb()
    model.train()

    model.train()
    step = 0
    for epoch in range(1025):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        for batch in train_loader:
            step += 1
            inputs = batch['input_ids'].to(rank)
            labels = batch['labels'].to(rank)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 100 == 0 and is_main_process():
                model.eval()
                val_batch = next(iter(val_loader))
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                with torch.no_grad():
                    val_outputs = model(**val_batch)
                    val_loss = val_outputs.loss
                    wandb.log({
                        "Step": step,
                        "Train Loss": loss.item(),
                        "Validation Loss": val_loss.item(),
                        "Epoch": epoch
                    })
                model.train()
        if is_main_process() and epoch in [0, 2, 5] or epoch % 64 == 0:
            path = f"./esm_{epoch}.pth"
            state = {'model': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     }
            torch.save(state, path)

    wandb.finish()
    cleanup()
    print("Training completed for rank: ", rank)


def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
