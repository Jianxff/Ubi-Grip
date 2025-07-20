from .mano_wrapper import MANO
from .hamer import HAMER
from .discriminator import Discriminator

from ..utils.download import cache_url

from pathlib import Path
import os

def download_models(folder):
    folder = Path(folder)
    if not folder.exists():
        os.makedirs(folder, exist_ok=True)
    output_file = folder / "hamer_demo_data.tar.gz"
    if not output_file.exists():
        url = "https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz"
        cache_url(url, output_file)
        os.system(f"tar -xvf {output_file}")


def load_hamer(checkpoint_path, mano_model_dir):
    from ..configs import get_config
    model_cfg = get_config(mano_model_dir)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
