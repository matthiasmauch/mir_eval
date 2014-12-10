# CREATED:2014-12-08 by Matthias Mauch <mail@matthiasmauch.net>
'''


'''

import numpy as np
import scipy.interpolate
import collections
import warnings
from . import util

import mir_eval

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


def rasterize_notes(interval, value, max_time=0.0, hopsize=0.01):
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
        - max_time : float
            Maximum timestamp assumed. If zero, then the largest interval
            offset timestamp is used.
        - hopsize : float
            Period in seconds at which to sample the notes

    :returns:
        - values : np.ndarray, shape=(n_events, 1)
            Note values sampled at a regular hopsize intervals.

    '''

    interval, value = remove_zero_notes(interval, value)

    max_time = max([interval.max(), max_time])
    print max_time
    n_note = len(value)

    n_frame = max_time / hopsize + 1
    times = np.arange(n_frame) * hopsize

    values = np.zeros((n_frame, 1))

    for i_note in range(n_note):
        index = np.logical_and(times >= interval[i_note, 0],  
                               times <  interval[i_note, 1])
        if max(values[index]) > 0:
            raise # overlapping notes!
        else:
            values[index] = value[i_note]
    return times, values


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

    max_time = max(ref_interval.max(), est_interval.max())
    ref_times, ref_midi = rasterize_notes(ref_interval, ref_midi, max_time)
    est_times, est_midi = rasterize_notes(est_interval, est_midi, max_time)

    # Compute metrics
    scores = collections.OrderedDict()

    (scores['Voicing Recall'],
     scores['Voicing False Alarm']) = util.filter_kwargs(mir_eval.melody.voicing_measures,
                                                         ref_midi>0,
                                                         est_midi>0, **kwargs)

    scores['Raw Pitch Accuracy'] = util.filter_kwargs(mir_eval.melody.raw_pitch_accuracy,
                                                      ref_midi>0, ref_midi*100,
                                                      est_midi>0, est_midi*100,
                                                      **kwargs)

    return scores
