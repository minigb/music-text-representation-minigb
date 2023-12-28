# Modified the provided demo.ipynb
import torch
from torch import nn
import numpy as np
import pandas as pd
import warnings
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import json

from mtr.utils.demo_utils import get_model
from utils.embeddings import audio_infer, text_infer, pre_extract_audio_embedding, pre_extract_text_embedding
from ..tag_to_music.dataset import MusicCaps, SongDescriber

warnings.filterwarnings(action='ignore')

msd_path = '/home/minhee/userdata/workspace/music-text-representation-minigb/dataset'


def load_embeddings(embs_path):
    embs_dict = torch.load(embs_path)
    ids = [k for k in embs_dict.keys()]
    embs = [embs_dict[k] for k in id]
    embs = torch.stack(embs)
    return embs, ids


def retrieve(model, tokenizer, text_list, audio_embs_path):

    meta_results = []
    for text in text_list:
        text_input = tokenizer(text, return_tensors="pt")['input_ids']
        with torch.no_grad():
            text_emb = model.encode_bert_text(text_input, None)
        
        audio_embs = nn.functional.normalize(audio_embs, dim=1)
        text_emb = nn.functional.normalize(text_emb, dim=1)
        logits = text_emb @ audio_embs.T
        ret_item = pd.Series(logits.squeeze(0).numpy(), index=id).sort_values(ascending=False)

        metadata = {}
        for idx, _id in enumerate(ret_item.sort_values(ascending=False).head(3).index):
            metadata[f'top{idx+1} music'] = _id

        meta_results.append(metadata)
    return meta_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type = str, required = True)
    parser.add_argument("--dataset_config_path", type = str, required = True)
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--tag_type", type = str, default = 'matching')
    args = parser.parse_args()

    # config
    config_path = Path(args.config_path)
    config = OmegaConf.load(config_path)
    model, tokenizer, _ = get_model(config.framework, config.text_type, config.text_rep)

    # dataset config
    dataset_config_path = Path(args.dataset_config_path)
    dataset_config = OmegaConf.load(dataset_config_path)
    
    # dataset
    if args.dataset == 'musiccaps':
        dataset = MusicCaps(dataset_config.musiccaps.csv_path)
        duration = [10 for _ in dataset.df.index]
    elif args.dataset == 'song_describer':
        dataset = SongDescriber(dataset_config.song_describer.csv_path)
        duration = [dataset[i]['duration'] for i in range(len(dataset))]
    else:
        raise NotImplementedError
    
    # audio embed
    audio_embs_path = Path(f'{args.dataset}_audio_embs.pt')
    if not audio_embs_path.exists():
        audio_embs_dict = pre_extract_audio_embedding(
            [dataset.get_identifier(i) for i in dataset.df.index],
            dataset.df['audio_path'],
            model,
            duration,)
        torch.save(audio_embs_dict, audio_embs_path)
    else:
        audio_embs_dict = torch.load(audio_embs_path)
    
    # tag
    tag_path = Path(dataset_config.tags_dir) / f'{args.tag_type}/{args.dataset}_tags.json'
    with open(tag_path, 'r') as f:
        tag_list = json.load(f)
    assert len(dataset) == len(tag_list), f'{len(dataset)} != {len(tag_list)}'
    dataset.df['tag'] = tag_list

    text_embs_path = Path(f'{args.dataset}_text_embs.pt')
    if not text_embs_path.exists():
        text_embs_dict = pre_extract_text_embedding(
            tag_list,
            model,
            tokenizer)
        torch.save(text_embs_dict, text_embs_path)
    else:
        text_embs_dict = torch.load(text_embs_path)

    

    




    print(retrieve(model, tokenizer, query))