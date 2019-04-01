import lyricwikia
import pandas as pd
import numpy as np
from torch.utils import data

class Dataset(data.Dataset):
    """
    Helper class for loading the dataset & getting the lyrics for each song
    """
    def __init__(self, path, result_path):
        self.path = path
        self.result_path = result_path
        self.lyrics_threeshold = 4
        self.read_dataset()


    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        raise NotImplementedError

    def summarize_dataset(self):
        print('Headers :            {}'.format(self.dataset.columns.values))
        print('Number of samples :  {}'.format(self.__len__()))
        print('Number of artists :  {}'.format(np.unique(self.dataset['artist_name']).shape[0]))

    def read_dataset(self):
        self.dataset = pd.read_csv(self.path)
    
    def api_result(self, status):
        if status == True:
            return 'Succes'

        return 'Fail'

    def get_lyrics(self):
        self.dataset['lyrics'] = '-'
        
        print('Starting the lyrics reader...')
        fails = 0
        for idx, data in self.dataset.iterrows():
            artist = data['artist_name']
            track = data['track_name']
            has_failed = True
            try:
                lyrics = lyricwikia.get_lyrics(data['artist_name'], data['track_name']).split('\n')
                lyrics.remove('')

                #get each line with its frequency
                lyrics, counts = np.unique(lyrics, return_counts=True)

                #sort the lyrics by their frequency
                count_sort_ind = np.argsort(-counts)

                #set the lyrics in the dataframe
                self.dataset.loc[idx, 'lyrics'] = '\n'.join(lyrics[count_sort_ind[1:1+self.lyrics_threeshold]])
            except:
                print('Couldn t get {} - {} at index {}'.format(artist, track, idx))
                fails = fails + 1
                has_failed = False

            print('{}. {} - {} : {}'.format(idx, artist, track, self.api_result(has_failed)))
            
            if idx == 10:
                break

        self.dataset.to_csv(self.result_path, index=False)

        print('{} samples failed to import'.format(fails))