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
    # mask = power > threshold
    # mask = power > np.max(power) - 65
    mask = power > np.max(power) - 50
    mask = 0.5 + quantile_weight * (mask - 0.5)
    mask = np.round(mask) # to have either 0 or 1 values
    if mask.dtype != 'float32':
        mask = np.float32(mask) # convert to float32
    return mask, sorted_power

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
    # vad = power > threshold
    vad = power > np.max(power) - 28 * 513
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
    ibm_labels, sorted_power = clean_speech_IBM(observations, quantile_fraction=ibm_quantile_fraction, quantile_weight=quantile_weight, eps=eps)
    ibm_labels = ibm_labels * vad_labels
    return ibm_labels, sorted_power

########################
#### Threshold-based IBM
########################

def _voiced_unvoiced_split_characteristic(number_of_frequency_bins):
    split_bin = 200
    transition_width = 99
    fast_transition_width = 5
    low_bin = 4
    high_bin = 500

    a = np.arange(0, transition_width)
    a = np.pi / (transition_width - 1) * a
    transition = 0.5 * (1 + np.cos(a))

    b = np.arange(0, fast_transition_width)
    b = np.pi / (fast_transition_width - 1) * b
    fast_transition = (np.cos(b) + 1) / 2

    transition_voiced_start = int(split_bin - transition_width / 2)
    voiced = np.ones(number_of_frequency_bins)

    # High Edge
    voiced[transition_voiced_start - 1: (
        transition_voiced_start + transition_width - 1)] = transition
    voiced[transition_voiced_start - 1 + transition_width: len(voiced)] = 0

    # Low Edge
    voiced[0: low_bin] = 0
    voiced[low_bin - 1: (low_bin + fast_transition_width - 1)] = \
        1 - fast_transition

    # Low Edge
    unvoiced = np.ones(number_of_frequency_bins)
    unvoiced[transition_voiced_start - 1: (
        transition_voiced_start + transition_width - 1)] = 1 - transition
    unvoiced[0: (transition_voiced_start)] = 0

    # High Edge
    unvoiced[high_bin - 1: (len(unvoiced))] = 0
    unvoiced[
    high_bin - 1: (high_bin + fast_transition_width - 1)] = fast_transition

    return (voiced, unvoiced)

def noise_aware_IBM(X, N,
                 threshold_unvoiced_speech=5,
                 threshold_voiced_speech=0,
                 threshold_unvoiced_noise=-10,
                 threshold_voiced_noise=-10,
                 low_cut=5,
                 high_cut=500):
    """Estimate an ideal binary mask given the speech and noise spectrum.
    :param X: speech signal in STFT domain with shape (frames, frequency-bins)
    :param N: noise signal in STFT domain with shape (frames, frequency-bins)
    :param threshold_unvoiced_speech:
    :param threshold_voiced_speech:
    :param threshold_unvoiced_noise:
    :param threshold_voiced_noise:
    :param low_cut: all values with frequency<low_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :param high_cut: all values with frequency>high_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :return: (speech mask, noise mask): tuple containing the two arrays,
        which are the masks for X and N
    """
    (voiced, unvoiced) = _voiced_unvoiced_split_characteristic(X.shape[-1])

    # calculate the thresholds
    threshold = threshold_voiced_speech * voiced + \
                threshold_unvoiced_speech * unvoiced
    threshold_new = threshold_unvoiced_noise * voiced + \
                    threshold_voiced_noise * unvoiced

    xPSD = X * X.conjugate()  # |X|^2 = Power-Spectral-Density

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    xPSD_threshold = xPSD / c
    c_new = np.power(10, (threshold_new / 10))
    xPSD_threshold_new = xPSD / c_new

    nPSD = N * N.conjugate()

    speechMask = (xPSD_threshold > nPSD)

    speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005))
    speechMask[..., 0:low_cut - 1] = 0
    speechMask[..., high_cut:len(speechMask[0])] = 0

    noiseMask = (xPSD_threshold_new < nPSD)

    noiseMask = np.logical_or(noiseMask, (xPSD_threshold_new < 0.005))
    noiseMask[..., 0: low_cut - 1] = 1
    noiseMask[..., high_cut: len(noiseMask[0])] = 1

    return (speechMask, noiseMask)


def threshold_IBM(X,
                 threshold_unvoiced_speech=5,
                 threshold_voiced_speech=0,
                 threshold_unvoiced_noise=-10,
                 threshold_voiced_noise=-10,
                 low_cut=5,
                 high_cut=500):
    """Estimate an ideal binary mask given the speech and noise spectrum.
    :param X: speech signal in STFT domain with shape (frames, frequency-bins)
    :param N: noise signal in STFT domain with shape (frames, frequency-bins)
    :param threshold_unvoiced_speech:
    :param threshold_voiced_speech:
    :param threshold_unvoiced_noise:
    :param threshold_voiced_noise:
    :param low_cut: all values with frequency<low_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :param high_cut: all values with frequency>high_cut are set to 0 in the
        speech mask ans set to 1 in the noise mask
    :return: (speech mask, noise mask): tuple containing the two arrays,
        which are the masks for X and N
    """
    (voiced, unvoiced) = _voiced_unvoiced_split_characteristic(X.shape[-1])

    # calculate the thresholds
    threshold = threshold_voiced_speech * voiced + \
                threshold_unvoiced_speech * unvoiced
    threshold_new = threshold_unvoiced_noise * voiced + \
                    threshold_voiced_noise * unvoiced

    xPSD = X * X.conjugate()  # |X|^2 = Power-Spectral-Density

    # each frequency is multiplied with another threshold
    c = np.power(10, (threshold / 10))
    xPSD_threshold = xPSD / c
    c_new = np.power(10, (threshold_new / 10))
    xPSD_threshold_new = xPSD / c_new

    # nPSD = N * N.conjugate()
    nPSD = 10

    speechMask = (xPSD_threshold > nPSD)

    speechMask = np.logical_and(speechMask, (xPSD_threshold > 0.005))
    speechMask[..., 0:low_cut - 1] = 0
    speechMask[..., high_cut:len(speechMask[0])] = 0

    return speechMask