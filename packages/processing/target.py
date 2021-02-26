"""
VAD
"""

import numpy as np

def clean_speech_IBM(observations,
                     quantile_fraction=0.98,
                     quantile_weight=0.999,
                     eps=1e-8):
    """ Calculate softened mask according to lorenz function criterion.
    :param observation: STFT of the the observed signal
    :param quantile_fraction: Fraction of observations which are rated down
    :param quantile_weight: Governs the influence of the mask
    :return: quantile_mask
    """
    power = abs(observations * observations.conj())
    power = 10 * np.log10(power + eps) # Smoother mask with log
    min_power = 10 * np.log10(eps) # Min value to make all values in sorted_power positive
    sorted_power = np.sort(power, axis=None)[::-1]
    sorted_power -= min_power # Subtract because min_power is negative
    lorenz_function = np.cumsum(sorted_power) / np.sum(sorted_power)
    # threshold = np.min(sorted_power[lorenz_function < quantile_fraction])
    threshold = sorted_power[lorenz_function < quantile_fraction][-1]
    threshold += min_power
    mask = power > threshold
    mask = 0.5 + quantile_weight * (mask - 0.5)
    mask = np.round(mask) # to have either 0 or 1 values
    if mask.dtype != 'float32':
        mask = np.float32(mask) # convert to float32
    return mask

def clean_speech_VAD(observations,
                     quantile_fraction=0.98,
                     quantile_weight=0.999,
                     eps=1e-8):
    """ Calculate softened mask according to lorenz function criterion.
    :param observation: STFT of the the observed signal
    :param quantile_fraction: Fraction of observations which are rated down
    :param quantile_weight: Governs the influence of the mask
    :return: quantile_mask
    """
    power = abs(observations * observations.conj())
    power = 10 * np.log10(power + eps) # Smoother mask with log
    min_power = 10 * np.log10(eps) * power.shape[0] # Min value to make all values in sorted_power positive
    power = power.sum(axis=0)
    sorted_power = np.sort(power, axis=None)[::-1]
    sorted_power -= min_power # Subtract because min_power is negative
    lorenz_function = np.cumsum(sorted_power) / np.sum(sorted_power)
    # threshold = np.min(sorted_power[lorenz_function < quantile_fraction])
    threshold = sorted_power[lorenz_function < quantile_fraction][-1]
    threshold += min_power
    vad = power > threshold
    vad = 0.5 + quantile_weight * (vad - 0.5)
    vad = np.round(vad) # to have either 0 or 1 values
    if vad.dtype != 'float32':
        vad = np.float32(vad) # convert to float32
    vad = vad[None]
    return vad

def noise_robust_clean_speech_VAD(observations,
                                  quantile_fraction_begin=0.93,
                                  quantile_fraction_end=0.99,
                                  quantile_weight=0.999,
                                  eps=1e-8):
    """
    Create VAD labels robust to noisy speech recordings.
    In particular, the labels are robust to noise occuring before speech.

    Args:
        observations ([type]): [description]
        quantile_fraction_begin (float, optional): [description]. Defaults to 0.93.
        quantile_fraction_end (float, optional): [description]. Defaults to 0.99.
        quantile_weight (float, optional): [description]. Defaults to 0.999.

    Returns:
        [type]: [description]
    """
    vad_labels = clean_speech_VAD(observations, quantile_fraction=quantile_fraction_begin, quantile_weight=quantile_weight, eps=eps)
    vad_labels = vad_labels[0]
    vad_labels_end = clean_speech_VAD(observations, quantile_fraction=quantile_fraction_end, quantile_weight=quantile_weight, eps=eps)
    vad_labels_end = vad_labels_end[0]
    indices_begin = np.nonzero(vad_labels)
    indices_end = np.nonzero(vad_labels_end)
    vad_labels[indices_begin[0][0]:indices_end[0][-1]] = (indices_end[0][-1]-indices_begin[0][0])*[1]
    vad_labels = vad_labels[None] # vad_labels.shape = (1, frames)
    return vad_labels

def noise_robust_clean_speech_IBM(observations,
                                  vad_quantile_fraction_begin=0.93,
                                  vad_quantile_fraction_end=0.99,
                                  ibm_quantile_fraction=0.999,
                                  quantile_weight=0.999,
                                  eps=1e-8):
    """
    Create IBM labels robust to noisy speech recordings using noise-robst VAD.
    In particular, the labels are robust to noise occuring before speech.

    Args:
        observations ([type]): [description]
        quantile_fraction_begin (float, optional): [description]. Defaults to 0.93.
        quantile_fraction_end (float, optional): [description]. Defaults to 0.99.
        quantile_weight (float, optional): [description]. Defaults to 0.999.

    Returns:
        [type]: [description]
    """
    vad_labels = noise_robust_clean_speech_VAD(observations,
                                               quantile_fraction_begin=vad_quantile_fraction_begin,
                                               quantile_fraction_end=vad_quantile_fraction_end,
                                               quantile_weight=quantile_weight,
                                               eps=eps)
    ibm_labels = clean_speech_IBM(observations, quantile_fraction=ibm_quantile_fraction, quantile_weight=quantile_weight, eps=eps)
    ibm_labels = ibm_labels * vad_labels
    return ibm_labels