# CREATED:2014-12-08 by Matthias Mauch <mail@matthiasmauch.net>
'''


'''

import numpy as np
import scipy.interpolate
import collections
import warnings
from . import util


def remove_zero_notes(interval, value):
    '''
    Takes a sequence of notes given by the time interval (in seconds) and
    a value, and removes those with 0 value.

    :parameters:
    - interval : np.ndarray, shape=(n_events, 2)
        Onset and offset time of each note.
    - value : np.ndarray shape=(n_events, 1) (or list of values)
        Array of pitches or frequencies.

    :returns:
    - outinterval : np.ndarray, shape=(n_events, 2)
        Onset and offset time of each note.
    - outvalue : np.ndarray shape=(n_events, 1)
        Array of pitches or frequencies.

    '''

    value = np.array(value)
    outinterval = interval[value>0,:]
    outvalue = value[value>0]
    return outinterval, outvalue


def rasterize_notes(interval, value, hopsize=0.01):
    '''
    Takes a sequence of notes given by the time interval (in seconds) and
    a value and turns them into a single list of frame-wise values sampled
    at hopsize intervals.

    :usage:
        TODO

    :parameters:
        - interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note
        - value : np.ndarray shape=(n_events, 1) (or list of values)
            Array of pitches or frequencies
        - hopsize : float
            Period in seconds at which to sample the notes

    :returns:
        - rasterized_notes : np.ndarray, shape=(n_events, 1)
            Note values sampled at a regular hopsize intervals.

    '''

    interval, value = remove_zero_notes(interval, value)

    n_note = len(value)
    n_frame = n_note / hopsize + 1
    t = np.arange(n_frame) * hopsize

    rasterized_notes = np.zeros((n_note, 1))

    for i_note in range(n_note):
        index = np.logical_and(t >= inverval[i_note, 1],  
                               t <  inverval[i_note, 2])
        if max(f0[index]) > 0:
            raise # overlapping notes!
        else:
            rasterized_notes[index] = value[i_note]
    return rasterized_notes


def evaluate(ref_interval, ref_midi, est_interval, est_midi, **kwargs):
    '''
    Evaluate two note transcriptions, where the first is treated as the
    reference (ground truth) and the second as the estimate to be evaluated
    (prediction).

    :usage: 
        TODO

    :parameters:
        - ref_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each reference note
        - ref_midi : np.ndarray
            Array of reference MIDI note values (not necessarily integer)
        - est_interval : np.ndarray, shape=(n_events, 2)
            Time of each estimated frequency value
        - est_midi : np.ndarray
            Array of estimated MIDI note values (not necessarily integer)
        - kwargs
            Additional keyword arguments which will be passed to the
            appropriate metric or preprocessing functions.

    :returns:
        - scores : dict
            Dictionary of scores, where the key is the metric name (str) and
            the value is the (float) score achieved.

    '''

    ref_vector = rasterized_notes(ref_interval, ref_midi)
    est_vector = rasterized_notes(est_interval, est_midi)

    # Compute metrics
    scores = collections.OrderedDict()

    # (scores['Voicing Recall'],
    #  scores['Voicing False Alarm']) = util.filter_kwargs(voicing_measures,
    #                                                      ref_voicing,
    #                                                      est_voicing, **kwargs)

    # scores['Raw Pitch Accuracy'] = util.filter_kwargs(raw_pitch_accuracy,
    #                                                   ref_voicing, ref_cent,
    #                                                   est_voicing, est_cent,
    #                                                   **kwargs)

    return scores
