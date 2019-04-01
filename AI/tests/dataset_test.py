import sys
sys.path.append('../')
from dataset import Dataset

def dataset_test(initial_path, result_path):
    ds = Dataset(initial_path, result_path)
    assert ds.__len__() == 3, 'Length error! Got {}, expected {}'.format(ds.__len__(), 3)
    
    ds.get_lyrics()
    assert ('lyrics' in ds.dataset) == True, 'Missing lyrics column'

    assert ds.dataset.loc[0, 'lyrics'] == '-', 'Invalid lyrics(0)'
    assert ds.dataset.loc[1, 'lyrics'] == '-', 'Invalid lyrics(1)'
    assert ds.dataset.loc[2, 'lyrics'] != '-', 'Invalid lyrics(2)'

    print('Test finished successfully!')