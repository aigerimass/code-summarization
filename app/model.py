import logging

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

from unixcoder_pipeline.model.model import Seq2Seq

logger = logging.getLogger(__name__)
# might be slow for the first time
_model = torch.jit.load('model_example.pt')

# set model to eval mode
_model.eval()

model_name_or_path = "microsoft/unixcoder-base"

_tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
_config = RobertaConfig.from_pretrained(model_name_or_path)
_config.is_decoder = True
_encoder = RobertaModel.from_pretrained(model_name_or_path, config=_config)

model = Seq2Seq(
    encoder=_encoder,
    decoder=_encoder,
    config=_config,
    beam_size=10,
    max_length=256,
    sos_id=_tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
    eos_id=_tokenizer.sep_token_id,
)
device = "cpu"

checkpoint_prefix = "pytorch_model.bin"
_model_to_load = model.module if hasattr(model, "module") else model
_model_to_load.load_state_dict(torch.load(checkpoint_prefix))


def predict(
    context: str,
) -> str:
    tokens_ids = model.tokenize([context], max_length=512, mode="<encoder-decoder>")
    source_ids = torch.tensor(tokens_ids).to(device)
    prediction_ids = model.generate(source_ids)
    predictions = model.decode(prediction_ids)
    return [x.replace("<mask0>", "").strip() for x in predictions[0]][0]
