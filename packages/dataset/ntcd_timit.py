"""From towardsdatascience.com:

Data ingestion such as retrieving 
data from CSV, relational database, NoSQL, Hadoop etc.
We have to retrieve data from multiple sources all the time
so we better to have a dedicated function for data retrieval.
"""
from glob import glob
import os
import pickle
import pathlib

#TODO: create clean speech dataset, noise-aware dataset, hybrid dataset --> store in process dataset with metadata (SNR, etc.)

"""
Speech-related
"""
def video_list(input_video_dir,
                dataset_type='train',
                labels='vad_labels',
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
            file_paths = sorted(glob(data_dir + '**/*' + '_' + labels + '_upsampled.h5',recursive=True))
            
            # file_paths = sorted(glob(data_dir + '**/*' + '_' + 'ntcd_proc' + '_' + labels + '_upsampled.h5',recursive=True))
        else:
            file_paths = sorted(glob(data_dir + '**/*' + '_' + labels + '.h5',recursive=True))

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
    mat_file_paths = sorted(glob(data_dir + '**/*.mat',recursive=True))

    # Remove input_speech_dir from file_paths
    file_paths = ['ntcd_timit/Clean/volunteers/' + path.split('/')[-2] + '/straightcam/' \
        + os.path.splitext(os.path.basename(path))[0] + '.wav' for path in mat_file_paths]

    # Output_file_path
    output_file_paths = []
    for mat_file_path in mat_file_paths:
        #Get the last 3 folders of noisy_file_path
        p = pathlib.Path(mat_file_path)
        p = str(pathlib.Path(*p.parts[-3:]))
        p = os.path.splitext(p)[0] + '.wav'

        output_file_path = os.path.join('ntcd_timit/Clean/' + p)

        output_file_paths.append(output_file_path)
        
    return file_paths, output_file_paths

def noisy_speech_dict(input_speech_dir,
                dataset_type='train',
                dataset_size='complete'):
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
    mat_file_paths = sorted(glob(data_dir + '**/*.mat',recursive=True))

    # Get the last folder + filename (w/o extension)
    input_file_shortpaths = [path.split('/')[-2] + '/straightcam/' \
        + os.path.splitext(os.path.basename(path))[0] + '.wav' for path in mat_file_paths]
    
    output_file_shortpaths = []
    for mat_file_path in mat_file_paths:
        p = pathlib.Path(mat_file_path)
        p = str(pathlib.Path(*p.parts[-3:]))
        p = os.path.splitext(p)[0] + '.wav'
        output_file_shortpaths.append(p)

    # # Noisy files
    # noisy_file_paths = sorted(glob(data_dir + '*/*/volunteers/**/*.wav',recursive=True))

    # List of noise types
    noise_types = ['Babble', 'Cafe', 'Car', 'LR', 'Street', 'White']

    # List of SNRs
    # snrs = ['-5', '0', '5', '10', '15', '20']
    snrs = ['-5', '0', '5']
    #TODO: snrs til -20dB

    if dataset_size == 'subset':
        # List of noise types
        noise_types = ['Babble']

        # List of SNRs
        snrs = ['-5']

    # Dict of raw noisy / processed noisy pairs
    noisy_input_output_pair_paths = {}

    for noise_type in noise_types:
        for snr in snrs:
            noisy_file_dir = os.path.join('ntcd_timit/u/drspeech/data/TCDTIMIT/Noisy_TCDTIMIT',
                                           noise_type,
                                           snr,
                                           'volunteers')
            
            # Input subset            
            subset_noisy_file_paths = [os.path.join(noisy_file_dir, file_shortpath)\
                for file_shortpath in input_file_shortpaths]
            
            # Output subset            
            output_noisy_file_dir = os.path.join('ntcd_timit',
                                            'Noisy',
                                           noise_type,
                                           snr)

            subset_output_noisy_file_paths = [os.path.join(output_noisy_file_dir, file_shortpath)\
                for file_shortpath in output_file_shortpaths]

            # Extend dict
            noisy_input_output_pair_paths.update(dict(zip(subset_noisy_file_paths, subset_output_noisy_file_paths)))

    return noisy_input_output_pair_paths


# dict mapping noisy file to clean file
def noisy_clean_pair_dict(input_speech_dir,
                dataset_type='train',
                dataset_size='complete'):
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
    mat_file_paths = sorted(glob(data_dir + '**/*.mat',recursive=True))

    # Get the last folder + filename (w/o extension)
    input_file_shortpaths = [path.split('/')[-2] + '/straightcam/' \
        + os.path.splitext(os.path.basename(path))[0] + '.wav' for path in mat_file_paths]
    
    output_file_shortpaths = []
    for mat_file_path in mat_file_paths:
        p = pathlib.Path(mat_file_path)
        p = str(pathlib.Path(*p.parts[-3:]))
        p = os.path.splitext(p)[0] + '.wav'
        output_file_shortpaths.append(p)

    # List of noise types
    noise_types = ['Babble', 'Cafe', 'Car', 'LR', 'Street', 'White']

    # List of SNRs
    # snrs = ['-5', '0', '5', '10', '15', '20']
    snrs = ['-5', '0', '5']
    #TODO: snrs til -20dB

    # Clean dir
    clean_file_dir = 'ntcd_timit/Clean/'

    ### Training data
    if dataset_type == 'train':
        clean_file_dir += 'train/'

    ### Validation data
    if dataset_type == 'validation':
        clean_file_dir += 'dev/'

    ### Test data
    if dataset_type == 'test':
        clean_file_dir += 'test/'

    if dataset_size == 'subset':
        # List of noise types
        noise_types = ['Babble']

        # List of SNRs
        snrs = ['-5']

    # Dict of noisy / clean pairs
    noisy_clean_pair_paths = {}

    for noise_type in noise_types:
        for snr in snrs:
            noisy_file_dir = os.path.join('ntcd_timit/u/drspeech/data/TCDTIMIT/Noisy_TCDTIMIT',
                                           noise_type,
                                           snr,
                                           'volunteers')
            
            # Noisy subset
            subset_noisy_file_paths = [os.path.join(noisy_file_dir, file_shortpath)\
                for file_shortpath in input_file_shortpaths]

            # Clean subset
            subset_clean_file_paths = [clean_file_dir + path.split('/')[-3] + '/' \
                + os.path.basename(path) for path in subset_noisy_file_paths]

            # Extend dict
            noisy_clean_pair_paths.update(dict(zip(subset_noisy_file_paths, subset_clean_file_paths)))

    return noisy_clean_pair_paths


# dict mapping processed noisy file processed to clean file
def proc_noisy_clean_pair_dict(input_speech_dir,
                dataset_type='train',
                dataset_size='complete',
                labels='vad_labels'):
    """
    Create clean speech + clean speech VAD

    Args:
        dataset_type (str, optional): [description]. Defaults to 'training'.

    Raises:
        ValueError: [description]

    Return:
        Audio_files (list)
    """

    # Clean dir
    clean_file_dir = input_speech_dir + 'ntcd_timit/Clean/'

    ### Training data
    if dataset_type == 'train':
        clean_file_dir += 'train/'

    ### Validation data
    if dataset_type == 'validation':
        clean_file_dir += 'dev/'

    ### Test data
    if dataset_type == 'test':
        clean_file_dir += 'test/'

    # List of files
    clean_file_paths = sorted(glob(clean_file_dir + '**/*.h5',recursive=True))

    # Get shortpaths
    file_shortpaths = []
    for clean_file_path in clean_file_paths:
        p = pathlib.Path(clean_file_path)
        p = str(pathlib.Path(*p.parts[-3:]))
        p = os.path.splitext(p)[0] # Remove extension
        p = p.replace('_' + labels, '')
        p = p + '.wav'
        file_shortpaths.append(p)
    
    # List of noise types
    noise_types = ['Babble', 'Cafe', 'Car', 'LR', 'Street', 'White']

    # List of SNRs
    # snrs = ['-5', '0', '5', '10', '15', '20']
    snrs = ['-5', '0', '5']

    if dataset_size == 'subset':
        # List of noise types
        noise_types = ['Babble']

        # List of SNRs
        snrs = ['-5']

    # Dict of noisy / clean pairs
    noisy_clean_pair_paths = {}

    for noise_type in noise_types:
        for snr in snrs:
            # Output subset            
            noisy_file_dir = os.path.join('ntcd_timit',
                                          'Noisy',
                                           noise_type,
                                           snr)
            
            # Noisy subset
            subset_noisy_file_paths = [os.path.join(noisy_file_dir, file_shortpath)\
                                for file_shortpath in file_shortpaths]

            # Extend dict
            noisy_clean_pair_paths.update(dict(zip(subset_noisy_file_paths, clean_file_paths)))

    return noisy_clean_pair_paths