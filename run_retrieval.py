# import sys
# sys.path.insert(0, '/home/minhee/userdata/workspace/')

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
from tqdm import tqdm
import os

from mtr.utils.demo_utils import get_model
from utils.embeddings import audio_infer, text_infer, pre_extract_audio_embedding, pre_extract_text_embedding
from utils.dataset import MusicCaps, SongDescriber

warnings.filterwarnings(action='ignore')

msd_path = '/home/minhee/userdata/workspace/music-text-representation-minigb/dataset'


def load_embeddings(embs_path):
    embs_dict = torch.load(embs_path)
    ids = [k for k in embs_dict.keys()]
    embs = [embs_dict[k] for k in id]
    embs = torch.stack(embs)
    return embs, ids


def get_sim(audio_embs_dict, text_emb):
    audio_embs = torch.stack([audio_embs_dict[k] for k in audio_embs_dict.keys()])
    audio_id = [k for k in audio_embs_dict.keys()]
    
    audio_embs = nn.functional.normalize(audio_embs, dim=1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)
    logits = text_emb @ audio_embs.T
    return logits

    ret_item = pd.Series(logits.squeeze(0).numpy(), index=audio_id).sort_values(ascending=False)

    # metadata = {}
    # for idx, _id in enumerate(ret_item.sort_values(ascending=False).head(3).index):
    #     metadata[f'top{idx+1} music'] = _id

    # # meta_results.append(metadata)
    # meta_results = metadata
    # return meta_results
    return ret_item


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
    dataset_name = args.dataset_name
    if dataset_name == 'musiccaps':
        dataset = MusicCaps(dataset_config.musiccaps.csv_path)
        audio_dir = Path(dataset_config.musiccaps.audio)
        duration_list = [9.91 for _ in dataset.df.index]
    elif dataset_name == 'song_describer':
        dataset = SongDescriber(dataset_config.song_describer.csv_path)
        audio_dir = Path(dataset_config.song_describer.audio)
        duration_list = [dataset[i]['duration'] for i in range(len(dataset))]
    else:
        raise NotImplementedError
    
    # audio embed
    audio_embs_path = Path(f'{dataset_name}_audio_embs.pt')
    if not audio_embs_path.exists():
        audio_embs_dict = pre_extract_audio_embedding(
            [dataset.get_identifier(i) for i in range(len(dataset))],
            [audio_dir / dataset[i]['path'] for i in range(len(dataset))],
            model,
            duration_list,)
        torch.save(audio_embs_dict, audio_embs_path)
    else:
        audio_embs_dict = torch.load(audio_embs_path)
    
    # tag
    tag_path = Path(dataset_config.tags_dir) / f'{args.tag_type}/{dataset_name}_tags.json'
    with open(tag_path, 'r') as f:
        tag_list = json.load(f)
    tag_list = [tag_list[i] for i in list(dataset.df.index)]
    assert len(dataset) == len(tag_list), f'{len(dataset)} != {len(tag_list)}'
    dataset.df['tag'] = tag_list

    text_embs_path = Path(f'{dataset_name}_text_embs.pt')
    if not text_embs_path.exists():
        text_embs_dict = pre_extract_text_embedding(
            set([tag for tags in tag_list for tag in tags ]),
            model,
            tokenizer)
        torch.save(text_embs_dict, text_embs_path)
    else:
        text_embs_dict = torch.load(text_embs_path)

    # query, get similarity
    sim_dir = Path(f'sim/{dataset_name}')
    if sim_dir.exists():
        raise NotImplementedError
        # ret_item_list = pd.read_csv(result_save_path)
        # ret_item_list = [ret_item_list.iloc[i].dropna() for i in range(len(ret_item_list))]
        # ret_item_list = [pd.Series(ret_item_list[i].values, index=ret_item_list[i].index) for i in range(len(ret_item_list))]
    else:
        os.makedirs(sim_dir, exist_ok=True)
        ret_item_list = []
        for i, cur_tag in tqdm(enumerate(tag_list)): # cur_tag is a list of string tags
            query_list = cur_tag + [dataset[i]['caption']]
            sim_result = {}

            identifier = dataset.get_identifier(i)
            result_save_path = sim_dir / f'{identifier}.pt'
            
            for query in query_list:
                if query in text_embs_dict.keys():
                    text_emb = text_embs_dict[query]
                else:
                    text_emb = text_infer(query, model, tokenizer)

                sim = get_sim(audio_embs_dict, text_emb)
        
                sim_result[query] = sim
            
            torch.save(sim_result, result_save_path)

    # load
    # for i in range(len(dataset)):
    #     identifier = dataset.get_identifier(i)
    #     result_save_path = sim_dir / f'{identifier}.pt'
    #     sim_result = torch.load(result_save_path)
    #     print(sim_result)
            
    #         sim_result_all = torch.stack(sim_result_list, axis=0)

    #         # average all
    #         sim_avg = sim_result_all.mean(axis=0)
    #         ret_item = pd.Series(sim_avg.squeeze(0).numpy(), index=audio_embs_dict.keys()).sort_values(ascending=False)
    #         ret_item_list.append(ret_item)

    #     # save
    #     ret_item_list = pd.DataFrame(ret_item_list)
    #     ret_item_list.to_csv(result_save_path)

    # # recall at 10
    # for k in [10]:
    #     recall = 0
    #     for i in range(len(dataset)):
    #         if dataset.get_identifier(i) in ret_item_list[i].head(k).index:
    #             recall += 1
    #         print(f'recall@{k}: {recall / len(tag_list)}')
    #         break