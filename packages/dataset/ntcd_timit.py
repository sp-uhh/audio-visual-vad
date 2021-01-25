"""From towardsdatascience.com:

Data ingestion such as retrieving 
data from CSV, relational database, NoSQL, Hadoop etc.
We have to retrieve data from multiple sources all the time
so we better to have a dedicated function for data retrieval.
"""
from glob import glob
import os
import pickle

#TODO: create clean speech dataset, noise-aware dataset, hybrid dataset --> store in process dataset with metadata (SNR, etc.)

"""
Speech-related
"""
def video_list(input_video_dir,
                dataset_type='train',
                upsampled=False):
    """
    Create clean speech + clean speech VAD

    Args:
        dataset_type (str, optional): [description]. Defaults to 'training'.

    Raises:
        ValueError: [description]

    Return:
        Audio_files (list)
    """
    
    data_dir = input_video_dir + 'ntcd_timit/matlab_raw/'

    ### Training data
    if dataset_type == 'train':
        data_dir += 'train/'

    ### Validation data
    if dataset_type == 'validation':
        data_dir += 'dev/'

    ### Test data
    if dataset_type == 'test':
        data_dir += 'test/'

    # List of files
    file_paths = sorted(glob(data_dir + '**/*.mat',recursive=True))
    if not file_paths:
        if upsampled:
            file_paths = sorted(glob(data_dir + '**/*_upsampled.h5',recursive=True))
        else:
            file_paths = sorted(glob(data_dir + '**/*.h5',recursive=True))

    # Remove input_video_dir from file_paths
    file_paths = [os.path.relpath(path, input_video_dir) for path in file_paths]

    return file_paths

def speech_list(input_speech_dir,
                dataset_type='train'):
    """
    Create clean speech + clean speech VAD

    Args:
        dataset_type (str, optional): [description]. Defaults to 'training'.

    Raises:
        ValueError: [description]

    Return:
        Audio_files (list)
    """
    
    data_dir = input_speech_dir + 'ntcd_timit/matlab_raw/'

    ### Training data
    if dataset_type == 'train':
        data_dir += 'train/'

    ### Validation data
    if dataset_type == 'validation':
        data_dir += 'dev/'

    ### Test data
    if dataset_type == 'test':
        data_dir += 'test/'

    # List of files
    file_paths = sorted(glob(data_dir + '**/*.mat',recursive=True))

    # Remove input_speech_dir from file_paths
    file_paths = ['ntcd_timit/Clean/volunteers/' + path.split('/')[-2] + '/straightcam/' \
        + os.path.splitext(os.path.basename(path))[0] + '.wav' for path in file_paths]

    return file_paths