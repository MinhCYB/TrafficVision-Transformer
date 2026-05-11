def get_b16_config(config):
    """ ViT-B/16 configuration """
    config.patch_size = 16
    config.emb_dim = 768
    config.mlp_dim = 1536
    config.num_heads = 6
    config.num_layers = 2
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def get_b32_config(config):
    """ ViT-B/32 configuration """
    config = get_b16_config(config)
    config.patch_size = 32
    return config

def get_l16_config(config):
    """ ViT-L/16 configuration """
    config.patch_size = 16
    config.emb_dim = 1024
    config.mlp_dim = 4096
    config.num_heads = 16
    config.num_layers = 24
    config.attn_dropout_rate = 0.0
    config.dropout_rate = 0.1
    return config

def get_l32_config(config):
    """ Vit-L/32 configuration """
    config = get_l16_config(config)
    config.patch_size = 32
    return config