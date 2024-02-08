import logging
import mindspore as ms
from mindspore.amp import auto_mixed_precision

from src.utils.misc import lower_config
from src.models import LoFTR
from src.losses.loftr_loss import LoFTRLoss

_logger = logging.getLogger(__name__)


def build_model(config, pretrained_ckpt=None, **kwargs):
    _config = lower_config(config)
    if 'training_mode' in kwargs:
        loss = LoFTRLoss(_config)
        model = LoFTR(config=_config['loftr'], loss=loss)
    else:
        model = LoFTR(config=_config['loftr'])

    if pretrained_ckpt:
        ms.load_checkpoint(pretrained_ckpt, model)
        _logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")

    if 'amp_level' in kwargs:
        auto_mixed_precision(model, amp_level=kwargs["amp_level"])
    return model
