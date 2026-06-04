import argparse
from models.utils import process_config


# ─── Model arch sub-configs ───────────────────────────────────────────────────

_ARCH_CONFIGS = {
    'b16': dict(patch_size=16, emb_dim=768,  mlp_dim=1536, num_heads=6,  num_layers=2,  attn_dropout_rate=0.0, dropout_rate=0.1),
    'b32': dict(patch_size=32, emb_dim=768,  mlp_dim=1536, num_heads=6,  num_layers=2,  attn_dropout_rate=0.0, dropout_rate=0.1),
    'l16': dict(patch_size=16, emb_dim=1024, mlp_dim=4096, num_heads=16, num_layers=24, attn_dropout_rate=0.0, dropout_rate=0.1),
    'l32': dict(patch_size=32, emb_dim=1024, mlp_dim=4096, num_heads=16, num_layers=24, attn_dropout_rate=0.0, dropout_rate=0.1),
    'h14': dict(patch_size=14, emb_dim=1280, mlp_dim=5120, num_heads=16, num_layers=32, attn_dropout_rate=0.0, dropout_rate=0.1),
}


def _apply_arch(config):
    for k, v in _ARCH_CONFIGS[config.model_arch].items():
        setattr(config, k, v)
    return config


def _print_config(config):
    print('----------------- Config ---------------')
    for k, v in sorted(vars(config).items()):
        print('{:>35}: {:<30}'.format(str(k), str(v)))
    print('----------------- End -------------------')


# ─── Train config ─────────────────────────────────────────────────────────────

def get_train_config():
    parser = argparse.ArgumentParser("TrafficVision-Transformer — Anomaly Detection Training")

    # Experiment
    parser.add_argument("--exp-name",   type=str,   default="anomaly_ste_tte")
    parser.add_argument("--dataset",    type=str,   default="UIT-ADrone")
    parser.add_argument("--train",      type=int,   default=0, help="1: Train, 0: Eval/Inference")
    parser.add_argument("--n-gpu",      type=int,   default=1)
    parser.add_argument("--tensorboard", default=False, action='store_true')

    # Model
    parser.add_argument("--model-arch", type=str, default="b16",
                        choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--image-size", type=int, default=384, choices=[224, 256, 384])
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1,
                        help="[UNUSED in anomaly detection] kept for compatibility")

    # Data
    parser.add_argument("--data-dir",    type=str, default='../../UIT-ADrone')
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    # Training
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--wd",           type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int,   default=500)
    parser.add_argument("--train-steps",  type=int,   default=38000)

    # Checkpoint
    parser.add_argument("--checkpoint-path", type=str,
                        default="./experiments_andt_ADrone_STE_TTE/checkpoints/best.pth")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Thư mục gốc lưu checkpoint/result. Mặc định dùng ./experiments/")

    # Anomaly detection specific
    parser.add_argument("--anomaly-threshold", type=float, default=None,
                        help="MSE threshold for anomaly detection (auto-computed as mean+std if None)")

    config, _ = parser.parse_known_args()
    config = _apply_arch(config)
    process_config(config)
    _print_config(config)
    return config


# ─── Eval config ──────────────────────────────────────────────────────────────

def get_eval_config():
    parser = argparse.ArgumentParser("TrafficVision-Transformer — Evaluation")
    parser.add_argument("--model-arch",      type=str, default="b16", choices=['b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--image-size",      type=int, default=384, choices=[224, 256, 384])
    parser.add_argument("--batch-size",      type=int, default=4)
    parser.add_argument("--num-workers",     type=int, default=4)
    parser.add_argument("--data-dir",        type=str, default='../../UIT-ADrone')
    parser.add_argument("--num-classes",     type=int, default=1,
                        help="[UNUSED in anomaly detection] kept for compatibility")
    config, _ = parser.parse_known_args()
    config = _apply_arch(config)
    return config