# Modified the provided demo.ipynb
import torch
from torch import nn
import numpy as np
import pandas as pd
import warnings
import argparse
from pathlib import Path
from omegaconf import OmegaConf

from mtr.utils.demo_utils import get_model
from mtr.utils.eval_utils import _text_representation
from mtr.utils.audio_utils import load_audio, STR_CH_FIRST

warnings.filterwarnings(action='ignore')

msd_path = '/home/minhee/userdata/workspace/music-text-representation-minigb/dataset'


def load_embeddings(framework, text_type, text_rep):
    ecals_test = torch.load(f"mtr/{framework}/exp/transformer_cnn_cf_mel/{text_type}_{text_rep}/audio_embs.pt")
    msdid = [k for k in ecals_test.keys()]
    audio_embs = [ecals_test[k] for k in msdid]
    audio_embs = torch.stack(audio_embs)
    return audio_embs, msdid

def pre_extract_audio_embedding(id_list, audio_path_list, model, duration, sr=16000):
    assert duration is not None, "audio duration must be specified"

    audio_embs_dict = {}
    for id, audio_path in zip(id_list, audio_path_list):
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
        audio_embs_dict[id] = audio_embs
    
    return audio_embs_dict



def pre_extract_text_embedding(id_list, text_list, model, tokenizer) -> dict:
    text_embs_dict = {}
    for id, text in zip(id_list, text_list):
        text_input = tokenizer(text, return_tensors="pt")['input_ids']
        with torch.no_grad():
            text_embs = model.encode_bert_text(text_input, None)
        text_embs_dict[id] = text_embs
    
    return text_embs_dict


def retrieve(framework, text_type, text_rep, query_list):
    model, tokenizer, _ = get_model(framework=framework, text_type=text_type, text_rep=text_rep)
    
    # get audio embedding info
    audio_embs, msdid = pre_extract_audio_embedding(framework, text_type, text_rep)
    

    meta_results = []
    for query in query_list:
        text_input = tokenizer(query, return_tensors="pt")['input_ids']
        with torch.no_grad():
            text_embs = model.encode_bert_text(text_input, None)
        
        audio_embs = nn.functional.normalize(audio_embs, dim=1)
        text_embs = nn.functional.normalize(text_embs, dim=1)
        logits = text_embs @ audio_embs.T
        ret_item = pd.Series(logits.squeeze(0).numpy(), index=msdid).sort_values(ascending=False)

        metadata = {}
        for idx, _id in enumerate(ret_item.sort_values(ascending=False).head(3).index):
            metadata[f'top{idx+1} music'] = _id

        meta_results.append(metadata)
    return meta_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required = True)
    
    args = parser.parse_args()
    config_path = Path(args.config_path)
    assert config_path.exists(), f"{config_path} does not exist."
    config = OmegaConf.load(config_path)


    tag_query = "banjo"
    query = [tag_query]

    framework = config.framework
    text_type = config.text_type
    text_rep = config.text_rep

    print(retrieve(framework, text_type, text_rep, query))