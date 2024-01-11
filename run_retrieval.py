# Modified the provided demo.ipynb
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import json
from tqdm import tqdm
import os

from mtr.utils.demo_utils import get_model
from utils.embeddings import audio_infer, text_infer, pre_extract_audio_embedding, pre_extract_text_embedding
from utils.dataset import MusicCaps, SongDescriber


def get_sim(audio_embs_dict, text_emb) -> torch.Tensor:
    audio_embs = torch.stack([audio_embs_dict[k] for k in audio_embs_dict.keys()])
    
    audio_embs = nn.functional.normalize(audio_embs, dim=1)
    text_emb = nn.functional.normalize(text_emb, dim=-1)
    logits = text_emb @ audio_embs.T
    
    return logits


def get_audio_embed_dict(audio_embs_path, dataset) -> dict:
    if not audio_embs_path.exists():
        os.makedirs(audio_embs_path.parent, exist_ok=True)
        audio_embs_dict = pre_extract_audio_embedding(
            [dataset.get_identifier(i) for i in range(len(dataset))],
            [audio_dir / dataset[i]['path'] for i in range(len(dataset))],
            model,)
        torch.save(audio_embs_dict, audio_embs_path)
    else:
        audio_embs_dict = torch.load(audio_embs_path)
    
    assert len(dataset) == len(audio_embs_dict), f'{len(dataset)} != {len(audio_embs_dict)}'
    return audio_embs_dict


def get_tag_list(tag_path, dataset) -> list:
    assert tag_path.exists(), f'{tag_path} does not exist'

    with open(tag_path, 'r') as f:
        tag_list = json.load(f)
    tag_list = [tag_list[i] for i in list(dataset.df.index)]
    
    assert len(dataset) == len(tag_list), f'{len(dataset)} != {len(tag_list)}'
    
    return tag_list


def get_text_embed_dict(text_embs_path, dataset) -> dict:
    if not text_embs_path.exists():
        os.makedirs(text_embs_path.parent, exist_ok=True)
        text_embs_dict = pre_extract_text_embedding(
            set([tag for tags in dataset.df['tag'] for tag in tags ]),
            model,
            tokenizer)
        torch.save(text_embs_dict, text_embs_path)
    else:
        text_embs_dict = torch.load(text_embs_path)
    
    assert len(text_embs_dict) == len(set([tag for tags in dataset.df['tag'] for tag in tags ])),\
        f'{len(text_embs_dict)} != {len(set([tag for tags in dataset.df["tag"] for tag in tags ]))}'

    return text_embs_dict


def sim_calculate_and_save(sim_dir, dataset, audio_embs_dict, text_embs_dict) -> None:
    if sim_dir.exists():
        assert len(os.listdir(sim_dir)) == len(dataset), f'{len(os.listdir(sim_dir))} != {len(dataset)}'
        return
    
    os.makedirs(sim_dir, exist_ok=True)
    for i, cur_tags in tqdm(enumerate(dataset.df['tag'])): # cur_tags is a list of string tags
        query_list = cur_tags + [dataset[i]['caption']]
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

    return


def rank_calculate_and_save(rank_dir, dataset, audio_embs_dict, sim_dir, portion_list) -> None:
    for portion in portion_list:
        rank_dict = {}
        if portion == 'random':
            # this will be set randomly in recall_at_k_calculate_and_save
            continue
        else:
            rank_result_path = Path(rank_dir/f'{portion}.json')
            if not rank_result_path.exists():
                os.makedirs(rank_result_path.parent, exist_ok=True)
                for i in tqdm(range(len(dataset))):
                    identifier = dataset.get_identifier(i)
                    sim_result_path = sim_dir / f'{identifier}.pt'
                    sim_result = torch.load(sim_result_path)

                    # stack all
                    sim_result = torch.stack(list(sim_result.values()), axis=0)
                        
                    # average
                    if portion == -1:
                        sim_avg = sim_result.mean(axis=0)
                    else:
                        sim_avg = portion * sim_result[:-1].mean(axis=0) + (1 - portion) * sim_result[-1]
                    ret_item = pd.Series(sim_avg.squeeze(0).numpy(), index=audio_embs_dict.keys()).sort_values(ascending=False)
                    rank = ret_item.index.get_loc(identifier)
                    rank_dict[str(identifier)] = rank
                
                with open(rank_result_path, 'w') as f:
                    json.dump(rank_dict, f, indent = 4)

            assert rank_result_path.exists()
            with open(rank_result_path, 'r') as f:
                rank_dict = json.load(f)
            assert len(dataset) == len(rank_dict), f'{len(dataset)} != {len(rank_dict)}'


def recall_at_k_calculate_and_save(rank_dir, dataset, k_list, portion_list, save_dir) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for portion in portion_list:
        rank_dict = {}
        if portion == 'random':
            for i in range(len(dataset)):
                identifier = dataset.get_identifier(i)
                rank_dict[str(identifier)] = np.random.randint(0, len(dataset) - 1)
        else:
            rank_result_path = Path(rank_dir/f'{portion}.json')
            assert rank_result_path.exists()
            with open(rank_result_path, 'r') as f:
                rank_dict = json.load(f)
        assert len(dataset) == len(rank_dict), f'{len(dataset)} != {len(rank_dict)}'

        recall_at_k_result = {k: {} for k in k_list}
        for k in k_list:
            count_retrieved = 0
            for identifier in rank_dict.keys():    
                if rank_dict[identifier] < k:
                    count_retrieved += 1
            
            save_path = Path(save_dir/f'{portion}.json')
            os.makedirs(save_dir, exist_ok=True)
            recall_at_k_result[k] = {'retrieved' : count_retrieved,
                                     'total' : len(dataset),
                                     'recall' : count_retrieved / len(dataset),
                                     'mean_rank' : np.mean(list(rank_dict.values())) + 1}
    
            with open(save_path, 'w') as f:
                json.dump(recall_at_k_result, f, indent = 4)


def get_dataset(dataset_config, dataset_name) -> tuple:
    if dataset_name == 'musiccaps':
        dataset = MusicCaps(dataset_config.musiccaps.csv_path)
        audio_dir = Path(dataset_config.musiccaps.audio)
    elif dataset_name == 'song_describer':
        dataset = SongDescriber(dataset_config.song_describer.csv_path)
        audio_dir = Path(dataset_config.song_describer.audio)
    else:
        raise NotImplementedError
    
    return dataset, audio_dir


def get_dirs(dataset_config, dataset_name, tag_type, dir_by_model_info, init_previous) -> tuple:
    dirs_to_save_process = {
        'embeds': Path(dir_by_model_info/'preprocessing'/dataset_name),
        'sim': Path(dir_by_model_info/'sim_result'/dataset_name/tag_type),
        'rank': Path('rank'/dir_by_model_info/dataset_name/tag_type),
        'result': Path('result'/dir_by_model_info/dataset_name/tag_type),
    }
    if init_previous:
        for dir_path in dirs_to_save_process.values():
            if dir_path.exists():
                os.rmdir(dir_path)

    tag_path = Path(dataset_config.tags_dir)/tag_type/f'{dataset_name}_tags.json'

    return dirs_to_save_process, tag_path

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", type = str, required = True)
    parser.add_argument("--framework", type = str, default = 'contrastive')
    parser.add_argument("--text_type", type = str, default = 'bert')
    parser.add_argument("--text_rep", type = str, required = True)

    # parser.add_argument("--dataset_dir", type = str, default = '../tag_to_music/dataset')
    parser.add_argument("--dataset_config_path", type = str, required = True)
    parser.add_argument("--dataset_name", type = str, required = True)
    parser.add_argument("--tag_type", type = str, default = 'matching')

    parser.add_argument("--init", type = bool, default = False)
    
    args = parser.parse_args()

    framework = args.framework
    text_type = args.text_type
    text_rep = args.text_rep
    dataset_config_path = Path(args.dataset_config_path)
    tag_type = args.tag_type
    dataset_name = args.dataset_name

    # model
    model, tokenizer, _ = get_model(framework, text_type, text_rep)
    dir_by_model_info = Path(f'{framework}/{text_type}/{text_rep}')

    # dataset
    dataset_config = OmegaConf.load(dataset_config_path)
    dataset, audio_dir = get_dataset(dataset_config, dataset_name)
    
    # init
    init_previous = args.init
    
    # path
    dirs, tag_path = get_dirs(dataset_config, dataset_name, tag_type, dir_by_model_info, init_previous)

    # calculate sim
    audio_embs_dict = get_audio_embed_dict(dirs['embeds'] / 'audio_embs.pt', dataset)
    dataset.df['tag'] = get_tag_list(tag_path, dataset)
    text_embs_dict = get_text_embed_dict(dirs['embeds'] / f'text_embs_{tag_type}.pt', dataset)
    sim_calculate_and_save(dirs['sim'], dataset, audio_embs_dict, text_embs_dict)

    # calculate rank
    # TODO(minigb): Take portion_list and k_list out to config or args
    portion_list = [-1, 0, 0.3, 0.5, 0.7, 'random'] # -1: just average, 0: caption only
    rank_calculate_and_save(dirs['rank'], dataset, audio_embs_dict, dirs['sim'], portion_list)

    # calculate recall@k
    k_list = [1, 5, 10]
    recall_at_k_calculate_and_save(dirs['rank'], dataset, k_list, portion_list, dirs['result'])