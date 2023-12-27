# Modified the provided code in README
import torch
import numpy as np

from mtr.utils.audio_utils import load_audio, STR_CH_FIRST


def audio_infer(audio_path, model, duration, sr=16000):
    assert duration is not None, "audio duration must be specified"

    audio, _ = load_audio(
            path=audio_path,
            ch_format= STR_CH_FIRST,
            sample_rate= sr,
            downmix_to_mono= True
    )

    input_size = int(duration * sr)
    hop = int(len(audio) // input_size)
    audio = np.stack([np.array(audio[i * input_size : (i + 1) * input_size]) for i in range(hop)]).astype('float32')
    audio_tensor = torch.from_numpy(audio)

    with torch.no_grad():
        z_audio = model.encode_audio(audio_tensor)
    audio_embs = z_audio.mean(0).detach().cpu()

    return audio_embs


def text_infer(text, model, tokenizer):
    text_input = tokenizer(text, return_tensors="pt")['input_ids']
    with torch.no_grad():
        text_embs = model.encode_bert_text(text_input, None)
    text_embs = text_embs.mean(0).detach().cpu()

    return text_embs


def pre_extract_audio_embedding(id_list, audio_path_list, model, duration, sr=16000) -> dict:
    assert duration is not None, "audio duration must be specified"

    audio_embs_dict = {}
    for id, audio_path in zip(id_list, audio_path_list):
        audio_embs = audio_infer(audio_path, model, duration, sr)
        audio_embs_dict[id] = audio_embs
    
    return audio_embs_dict


def pre_extract_text_embedding(text_list, model, tokenizer) -> dict:
    text_embs_dict = {}
    for text in text_list:
        # text can be treated as an identifier by itself
        text_embs = text_infer(text, model, tokenizer)
        text_embs_dict[text] = text_embs
    
    return text_embs_dict