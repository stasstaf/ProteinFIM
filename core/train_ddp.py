from model.language_model import LanguageModel, GPTConfig
from data.make_data import process_file, process_raw, rng, process_for_val_batch
import os.path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import wandb


print('loading data...')

if not os.path.isfile('data/train.txt'):
    train, val, test = process_raw("data/raw/AFDBv4_90.128-254.fasta")
    train.to_csv('data/train.txt', index=False, header=False)
    test.to_csv('data/test.txt', index=False, header=False)
    val.to_csv('data/val.txt', index=False, header=False)

train_data = process_file('data/train.txt')
val_data = process_file('data/val.txt')
val_data_df = process_for_val_batch('data/val.txt')

vocab = sorted(
    list(set("".join(train_data))) + ['0'])  # <PRE> = '@', <MID> = '#', <SUF> = '$', <EOS> = '.', <PAD> = '0'
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
encode = lambda s: torch.LongTensor([stoi[c] for c in s])
decode = lambda l: "".join([itos[i] for i in l])

print('data loaded successfully')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    wandb.login(key='353758ac65c9ac5ceab0c5b51ce078ea9176161d')
    wandb.init(project='diploma', entity='stasstaf')

    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def main(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cuda", rank)
    print(device)
    config = GPTConfig()
    model = LanguageModel(config)
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    batch_size = 512
    steps = 10000
    ctx_size = config.ctx_size

    def get_batch(split, rank, world_size):
        data = train_data if split == 'train' else val_data
        per_worker = len(data) // world_size
        start_ix = rank * per_worker
        end_ix = start_ix + per_worker
        ix = rng.integers(start_ix, min(end_ix, len(data) - ctx_size), size=batch_size)
        x = torch.stack([encode(data[i:i + ctx_size]) for i in ix])
        y = torch.stack([encode(data[i + 1:i + ctx_size + 1]) for i in ix])
        return x.to(device), y.to(device)

    def get_val_batch():
        data = val_data_df

        ix = rng.integers(len(data), size=batch_size)

        xs1 = []
        ys1 = []
        xs2 = []
        ys2 = []
        xs3 = []
        ys3 = []

        mask = []
        for i in ix:
            document = data[i][:ctx_size]
            n = len(document)
            idx1, idx2 = torch.randperm(n - 2)[:2] + 1
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1

            prefix, middle, suffix = document[:idx1], document[idx1:idx2], document[idx2:]
            fim_sample = '@' + prefix + '$' + suffix + '#' + middle

            sample_x1 = encode(fim_sample[:ctx_size])
            sample_y1 = encode(fim_sample[1:ctx_size + 1])

            fim_sample = prefix + middle

            sample_x2 = encode(fim_sample[:ctx_size])
            sample_y2 = encode(fim_sample[1:ctx_size + 1])

            mask.append(torch.tensor(len(prefix)))

            sample_x3 = encode(data[i][:ctx_size])
            sample_y3 = encode(data[i][1:ctx_size + 1])

            sample_x1 = F.pad(sample_x1, (0, max(0, ctx_size - len(sample_x1))), value=stoi['0'])
            sample_y1 = F.pad(sample_y1, (0, max(0, ctx_size - len(sample_y1))), value=stoi['0'])

            sample_x2 = F.pad(sample_x2, (0, max(0, ctx_size - len(sample_x2))), value=stoi['0'])
            sample_y2 = F.pad(sample_y2, (0, max(0, ctx_size - len(sample_y2))), value=stoi['0'])

            sample_x3 = F.pad(sample_x3, (0, max(0, ctx_size - len(sample_x3))), value=stoi['0'])
            sample_y3 = F.pad(sample_y3, (0, max(0, ctx_size - len(sample_y3))), value=stoi['0'])

            xs1.append(sample_x1)
            ys1.append(sample_y1)

            xs2.append(sample_x2)
            ys2.append(sample_y2)

            xs3.append(sample_x3)
            ys3.append(sample_y3)

        x1 = torch.stack(xs1).to(device)
        y1 = torch.stack(ys1).to(device)

        x2 = torch.stack(xs2).to(device)
        y2 = torch.stack(ys2).to(device)

        x3 = torch.stack(xs3).to(device)
        y3 = torch.stack(ys3).to(device)

        mask = torch.stack(mask).to(device)
        return x1, x2, x3, y1, y2, y3, mask

    optim = torch.optim.AdamW(ddp_model.parameters(), lr=3e-5)
    ddp_model.train()

    for step in range(steps):
        xb, yb = get_batch('train', rank, world_size)

        optim.zero_grad()
        _, loss = ddp_model(xb, yb)
        loss.backward()
        optim.step()

        if step % 100 == 0 and is_main_process():
            model.eval()
            with torch.no_grad():
                splits = {}
                for split in ['train', 'val']:
                    losses = torch.zeros(1)
                    for k in range(1):
                        X, y = get_batch(split, rank, world_size)
                        logits, loss = model(X, y)
                        losses[k] = loss.item()
                    splits[split] = losses.mean()
                x1, x2, x3, y1, y2, y3, ixs = get_val_batch()
                loss_1 = model.calculate_loss(x3, y3, mode='default')

                loss_2 = model.calculate_loss(x2, y2, mode='pms', indexes=ixs)

                loss_3 = model.calculate_loss(x1, y1, mode='default')

                loss_4 = model.calculate_loss(x1, y1, mode='psm')

                wandb.log({
                    "Step": step,
                    "Train Loss": splits['train'],
                    "Validation Loss": splits['val'],
                    "AR Loss": loss_1,
                    "AR Middle Loss": loss_2,
                    "FIM Loss": loss_3,
                    "FIM Middle Loss": loss_4
                })

    if is_main_process():
        print()
        print(f"Final loss: {loss.item()}")
        path = "./fim_gpt_last.pth"
        torch.save(ddp_model, path)
        wandb.finish()


    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)