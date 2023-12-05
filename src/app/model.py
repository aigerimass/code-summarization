import logging

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel  # type: ignore

from unixcoder_pipeline.model.model import Seq2Seq

logger = logging.getLogger(__name__)

model_name_or_path = "microsoft/unixcoder-base"

_tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
_config = RobertaConfig.from_pretrained(model_name_or_path)
_config.is_decoder = True
_encoder = RobertaModel.from_pretrained(model_name_or_path, config=_config)

_model = Seq2Seq(
    encoder=_encoder,
    decoder=_encoder,
    config=_config,
    beam_size=10,
    max_length=256,
    sos_id=_tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
    eos_id=_tokenizer.sep_token_id,
)
device = "cpu"

checkpoint_prefix = "src/app/pytorch_model.bin"
_model_to_load = _model.module if hasattr(_model, "module") else _model
_model_to_load.load_state_dict(torch.load(checkpoint_prefix, map_location=torch.device('cpu')))
_model = _model.to(torch.device('cpu'))
_model.eval()
print(torch.cuda.is_available())


def predict(context: str) -> str:
    max_source_length = 256
    tokens_ids = _tokenizer.tokenize(context)[: max_source_length]
    tokens_ids = (
        [_tokenizer.cls_token, "<encoder-decoder>", _tokenizer.sep_token, "<mask0>"]
        + tokens_ids
        + [_tokenizer.sep_token]
    )
    tokens_ids = _tokenizer.convert_tokens_to_ids(tokens_ids)
    padding_length = max_source_length - len(tokens_ids)
    tokens_ids += [_tokenizer.pad_token_id] * padding_length
    with torch.no_grad():
        preds = _model(torch.LongTensor([tokens_ids]).to(torch.device('cpu')))
        # convert ids to text
        for pred in preds:
            t = pred[0].cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[: t.index(0)]
            text = _tokenizer.decode(t, clean_up_tokenization_spaces=False)
            return text
    raise NotImplementedError("torch.no_grad() error")
