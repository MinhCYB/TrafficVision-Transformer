from models.vit_ste_tte import VisionTransformer
from engine.trainer import train_epoch, valid_epoch
from configs.train_config import get_train_config

import torch

def main():

    config = get_train_config()

    model = VisionTransformer(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        num_frames=config.num_frames
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr
    )

    print("Training setup OK")

if __name__ == "__main__":
    main()