# Modified the provided code in README
import torch
import numpy as np
from tqdm import tqdm

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
    hop = max(int(len(audio) // input_size), 1)
    audio = np.stack([np.array(audio[i * input_size : (i + 1) * input_size]) for i in range(hop)]).astype('float32')
    audio_tensor = torch.from_numpy(audio)

    with torch.no_grad():
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.to('cuda')
            model = model.to('cuda')
        z_audio = model.encode_audio(audio_tensor)
    audio_embs = z_audio.mean(0).detach().cpu()

    return audio_embs


def text_infer(text, model, tokenizer):
    text_input = tokenizer(text, return_tensors="pt")['input_ids']
    with torch.no_grad():
        if torch.cuda.is_available():
            text_input = text_input.to('cuda')
            model = model.to('cuda')
        text_embs = model.encode_bert_text(text_input, None)
    text_embs = text_embs.mean(0).detach().cpu()

    return text_embs


def pre_extract_audio_embedding(id_list, audio_path_list, model, duration=9.91, sr=16000) -> dict:
    audio_embs_dict = {}
    assert len(id_list) == len(audio_path_list), f'{len(id_list)} != {len(audio_path_list)}'

    print(f'Extracting audio embeddings for {len(id_list)} audio files')
    for id, audio_path in tqdm(list(zip(id_list, audio_path_list))):
        audio_embs = audio_infer(audio_path, model, duration, sr)
        audio_embs_dict[id] = audio_embs
    
    return audio_embs_dict


def pre_extract_text_embedding(text_list, model, tokenizer) -> dict:
    text_embs_dict = {}
    for text in tqdm(text_list):
        # text itself can be treated as an identifier
        text_embs = text_infer(text, model, tokenizer)
        text_embs_dict[text] = text_embs
    
    return text_embs_dict