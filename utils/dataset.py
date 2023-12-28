# import sys
# sys.path.insert(0, '/home/minhee/userdata/workspace/tag_to_music/utils')

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

from utils.base import NonOverridableMeta
    

class Dataset(ABC, metaclass = NonOverridableMeta):
    _non_overridable_methods_ = {'__getitem__', '__len__'}

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = pd.read_csv(self.data_path)
        # Here we assume that the dataset has 'caption' column.
        # Change this part later if needed.
        assert 'caption' in self.df.columns, "There is no 'caption' column in the dataset."
    

    def __getitem__(self, index):
        return self.df.iloc[index]
    

    def __len__(self):
        return len(self.df)
    

    @abstractmethod
    def get_identifier(self, index):
        pass


class MusicCaps(Dataset):
    def __init__(self, data_path, text_only = False):
        super().__init__(data_path)
        self._set_audio_path()
        self.text_only = text_only
        if not self.text_only:
            self._drop_unavailables()
        self._index_list = list(self.df.index)


    def __getitem__(self, index):
        assert index < len(self)
        new_index = self._index_list[index]
        return self.df.iloc[new_index]


    def get_identifier(self, index):
        return self[index]['ytid'] # string
    

    def _set_audio_path(self, audio_type = 'wav'):
        for i in range(len(self)):
            data = self[i]
            file_name = f"[{data['ytid']}]-[{data['start_s']}-{data['end_s']}]"
            self.df.loc[i, 'path'] = f'{file_name}.{audio_type}'


    def _drop_unavailables(self):
        audio_dir = self.data_path.parent / 'audio'
        for i in range(len(self)):
            file_name = self[i]['path']
            audio_file = audio_dir / file_name
            if not audio_file.exists():
                self.df.drop(i, inplace = True)


class SongDescriber(Dataset):
    def __init__(self, data_path):
        super().__init__(data_path)


    def get_identifier(self, index):
        return self[index]['track_id'] # number