# CREATED:2014-12-08 by Matthias Mauch <mail@matthiasmauch.net>
'''


'''

import numpy as np
import pandas as pd
import scipy.interpolate
import collections
import warnings
from .util import filter_kwargs
from .monophonic_note_trans import *


from mir_eval.melody import voicing_measures, raw_pitch_accuracy,\
                            raw_chroma_accuracy, overall_accuracy

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

    max_time = max([interval.max(), max_time])

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


def calculate_matches(interval_A, midi_A, interval_B, midi_B, 
                      midi_threshold=0.5, onset_threshold=0.05):
    '''
    Calculate matches between two sets of note tracks, according to Molina's
    proposed metrics.

    :parameters:
        - interval_A : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in note track A.
        - midi_A : np.ndarray
            Array of MIDI note values (not necessarily integer) 
            in note track A.
        - interval_B : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in note track B.
        - midi_B : np.ndarray
            Array of MIDI note values (not necessarily integer) 
            in note track A.

    :returns:
        - matched : pd.DataFrame
            Data frame containing the success rating (0 or 1) for Molina's
            matching criteria on each note of note track A.
    '''
    
    n_A = len(midi_A)

    matched = pd.DataFrame(0, index=range(n_A), 
                              columns=['COnPOff', 'COnP', 'COn', 'CN'])
    
    conpoff_matches = []
    conp_matches = []
    con_matches = []
    cn_matches = []

    for i, iv_A in enumerate(interval_A):
        offset_thresh = iv_A[0] + (iv_A[1]-iv_A[0]) * np.array([0.8, 1.2])

        for j, iv_B in enumerate(interval_B):
            is_overlap = iv_A[0] < iv_B[1] and iv_A[1] > iv_B[0]
            is_correct_pitch = abs(midi_A[i]-midi_B[j]) < midi_threshold
            is_matched_onset = abs(iv_A[0]-iv_B[0]) < onset_threshold
            is_matched_offset = iv_B[1] > offset_thresh[0] and iv_B[1] < offset_thresh[1]
            is_good_overlap = False
            if is_overlap:
                overlap = min(iv_A[1], iv_B[1]) - max(iv_A[0], iv_B[0])
                max_length = max(iv_A[1]-iv_A[0], iv_B[1]-iv_B[0])
                overlap_ratio = float(overlap)/max_length
                print overlap_ratio
                is_good_overlap = overlap_ratio > 0.5 and abs(midi_A[i]-midi_B[j]) < 0.1

            if is_matched_onset and is_correct_pitch and is_matched_offset:
                if not j in conpoff_matches:
                    matched.COnPOff[i] = 1
                    conpoff_matches.append(j)

            if is_matched_onset and is_correct_pitch:
                if not j in conp_matches:
                    matched.COnP[i] = 1
                    conp_matches.append(j)

            if is_matched_onset:
                if not j in con_matches:
                    matched.COn[i] = 1
                    con_matches.append(j)

            if is_good_overlap:
                if not j in cn_matches:
                    matched.CN[i] = 1
                    cn_matches.append(j)

    return matched

def calculate_match_metrics(ref_interval, ref_midi, est_interval, est_midi):
    '''
    Calculate matches between two sets of note tracks, according to Molina's
    proposed metrics.

    :parameters:
        - ref_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in reference note track.
        - ref_midi : np.ndarray
            Array of MIDI note values (not necessarily integer) 
            in reference note track.
        - est_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in estimated note track.
        - est_midi : np.ndarray
            Array of MIDI note values (not necessarily integer) 
            in estimated note track.

    :returns:
        - precision: pd.DataFrame
            Precision of all implemented metrics.
        - recall: pd.DataFrame
            Recall of all implemented metrics.
        - fmeasure: pd.DataFrame
            Harmonic mean of precision and recall.
    '''

    matched_est_to_ref = calculate_matches(ref_interval, ref_midi, 
                                           est_interval, est_midi)
    matched_ref_to_est = calculate_matches(est_interval, est_midi, 
                                           ref_interval, ref_midi)

    precision = matched_ref_to_est.mean(0)
    recall    = matched_est_to_ref.mean(0)
    fmeasure  = 2 * precision * recall / (precision + recall)

    return precision, recall, fmeasure

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

    ref_interval, ref_midi = remove_zero_notes(ref_interval, ref_midi)
    est_interval, est_midi = remove_zero_notes(est_interval, est_midi)

    ref_frame_times, ref_frame_midi = rasterize_notes(ref_interval, ref_midi,
                                                      max_time)
    est_frame_times, est_frame_midi = rasterize_notes(est_interval, est_midi,
                                                      max_time)

    # Compute metrics
    scores = collections.OrderedDict()

    (scores['Voicing Recall'],
     scores['Voicing False Alarm']) = filter_kwargs(voicing_measures,
                                                    ref_frame_midi>0,
                                                    est_frame_midi>0, **kwargs)

    scores['Raw Pitch Accuracy'] = filter_kwargs(raw_pitch_accuracy,
                                                 ref_frame_midi>0,
                                                 ref_frame_midi*100,
                                                 est_frame_midi>0,
                                                 est_frame_midi*100,
                                                 **kwargs)
    scores['Raw Chroma Accuracy'] = filter_kwargs(raw_chroma_accuracy,
                                                       ref_frame_midi>0,
                                                       ref_frame_midi*100,
                                                       est_frame_midi>0,
                                                       est_frame_midi*100,
                                                       **kwargs)

    scores['Overall Accuracy'] = filter_kwargs(overall_accuracy,
                                                    ref_frame_midi>0,
                                                    ref_frame_midi*100,
                                                    est_frame_midi>0,
                                                    est_frame_midi*100,
                                                    **kwargs)

    precision, recall, fmeasure = calculate_match_metrics(ref_interval, ref_midi,
                                                          est_interval, est_midi)
    scores['Precision COnPOff'] = precision.COnPOff
    scores['Recall COnPOff'] = recall.COnPOff
    scores['F Measure COnPOff'] = fmeasure.COnPOff

    scores['Precision COnP'] = precision.COnP
    scores['Recall COnP'] = recall.COnP
    scores['F Measure COnP'] = fmeasure.COnP

    scores['Precision COn'] = precision.COn
    scores['Recall COn'] = recall.COn
    scores['F Measure COn'] = fmeasure.COn

    scores['Precision CN'] = precision.CN
    scores['Recall CN'] = recall.CN
    scores['F Measure CN'] = fmeasure.CN

    return scores
