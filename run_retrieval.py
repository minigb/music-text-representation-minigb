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

warnings.filterwarnings(action='ignore')

msd_path = '/home/minhee/userdata/workspace/music-text-representation-minigb/dataset'


def load_embeddings(embs_path):
    embs_dict = torch.load(embs_path)
    ids = [k for k in embs_dict.keys()]
    embs = [embs_dict[k] for k in id]
    embs = torch.stack(embs)
    return embs, ids


def retrieve(framework, text_type, text_rep, query_list, audio_embs_path, text_embs_path):
    model, tokenizer, _ = get_model(framework=framework, text_type=text_type, text_rep=text_rep)
    
    # get audio embedding info
    audio_embs, id = load_embeddings(audio_embs_path) # need to fix this


    meta_results = []
    for query in query_list:
        text_input = tokenizer(query, return_tensors="pt")['input_ids']
        with torch.no_grad():
            text_embs = model.encode_bert_text(text_input, None)
        
        audio_embs = nn.functional.normalize(audio_embs, dim=1)
        text_embs = nn.functional.normalize(text_embs, dim=1)
        logits = text_embs @ audio_embs.T
        ret_item = pd.Series(logits.squeeze(0).numpy(), index=id).sort_values(ascending=False)

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