from transformers import DataCollatorForLanguageModeling, EsmForMaskedLM, EsmTokenizer, EsmConfig
from data.make_data import make_esm_dataset
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
import wandb


def setup_wandb():
    wandb.login(key='353758ac65c9ac5ceab0c5b51ce078ea9176161d')
    wandb.init(project='diploma', entity='stasstaf')


if __name__ == "__main__":
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm-1b")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        return_tensors='pt'
    )
    file_path = "data/raw/AFDBv4_90.128-254.fasta"
    train_dataset, val_dataset, _ = make_esm_dataset(file_path, tokenizer)
    batch_size = 192
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = EsmConfig.from_pretrained("facebook/esm-1b")
    cfg.max_position_embeddings = 256
    cfg.intermediate_size = 768
    cfg.hidden_size = 768
    cfg.num_attention_heads = 12
    cfg.num_hidden_layers = 12
    model = EsmForMaskedLM(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)

    setup_wandb()
    model.train()
    step = 0
    epochs = 10000
    for epoch in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                model.eval()
                val_batch = next(iter(val_loader))
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                with torch.no_grad():
                    val_outputs = model(**val_batch)
                    val_loss = val_outputs.loss

                wandb.log({
                    "Step": step,
                    "Train Loss": loss.item(),
                    "Validation Loss": val_loss.item()
                })
                model.train()
            if step % 25000 == 0:
                path = f"./esm_{step}.pth"
                state = {'model': model.module.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         }
                torch.save(state, path)

            step += 1

    wandb.finish()
