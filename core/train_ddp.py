from model.language_model import LanguageModel, GPTConfig
from data.make_data import process_file, process_raw, rng
import os.path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

if not os.path.isfile('../data/train.txt'):
    train, val, test = process_raw("../data/raw/AFDBv4_90.128-254.fasta")
    train.to_csv('../data/train.txt', index=False, header=False)
    test.to_csv('../data/test.txt', index=False, header=False)
    val.to_csv('../data/val.txt', index=False, header=False)

train_data = process_file('../data/train.txt')
val_data = process_file('../data/val.txt')

vocab = sorted(
    list(set("".join(train_data))) + ['0'])  # <PRE> = '@', <MID> = '#', <SUF> = '$', <EOS> = '.', <PAD> = '0'
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for i, c in enumerate(vocab)}
encode = lambda s: torch.LongTensor([stoi[c] for c in s])
decode = lambda l: "".join([itos[i] for i in l])


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
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
    steps = 1000
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

    optim = torch.optim.AdamW(ddp_model.parameters(), lr=3e-5)
    ddp_model.train()

    for step in range(steps):
        xb, yb = get_batch('train', rank, world_size)

        optim.zero_grad()
        _, loss = ddp_model(xb, yb)
        loss.backward()
        optim.step()

        if step % 100 == 0 and is_main_process():
            print(f"Step {step:4}: loss {loss.item():.5f}")

    if is_main_process():
        print()
        print(f"Final loss: {loss.item()}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)