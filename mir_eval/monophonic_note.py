# CREATED:2014-12-08 by Matthias Mauch <mail@matthiasmauch.net>
'''


'''

import numpy as np
import pandas as pd
import scipy.interpolate
import collections
import warnings
from .util import filter_kwargs

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

def pseudomanual_merge(ref_interval, ref_midi, est_interval, est_midi):
    '''
    Takes a reference note track and an estimated note track, then merges
    estimated notes that are overlapped by reference notes and counts the 
    number of interactions needed.

    :parameters:
        - ref_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in reference note track.
        - est_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in estimated note track.
        - est_midi : np.ndarray
            Array of MIDI note values (not necessarily integer) 
            in estimated note track.
    :returns:
        - new_interval : np.ndarray, shape=(n_events, 2)
            Intervals without the deleted notes.
        - new_midi : np.ndarray
            Array of MIDI note values of the non-deleted notes.
    '''

    n_est = len(est_midi)
    n_ref = len(ref_midi)
    time_overlaps, timepitch_overlaps = calculate_overlaps(ref_interval,
                                                           ref_midi,
                                                           est_interval,
                                                           est_midi)

    # make merge plan

    merge_plan = []
    last_est = 0

    for i_ref in range(n_ref):
        overlaps = [iv for iv in time_overlaps if 
                    iv[0]==i_ref and time_overlaps[iv][2] > 0.5]
        est_overlap_ind = [el[1] for el in overlaps]
        if est_overlap_ind:
            # print "aha", range(last_est, min(est_overlap_ind)+1)
            merge_plan.extend([el] for 
                              el in range(last_est, min(est_overlap_ind)))
            merge_plan.append(est_overlap_ind)
            last_est = max(est_overlap_ind) + 1
    merge_plan.extend([el] for el in range(last_est, n_est))

    # compose new notes from merge plan

    new_interval = list()
    new_midi = list()

    for group in merge_plan:
        dur = [ei[1]-ei[0] for ei in est_interval[group, :]]
        onset = min([ei[0] for ei in est_interval[group, :]])
        offset = max([ei[1] for ei in est_interval[group, :]])
        midi = est_midi[group][np.argmax(dur)]
        new_interval.append((onset, offset))
        new_midi.append(midi)

    return np.array(new_interval), np.array(new_midi)


def pseudomanual_delete(ref_interval, ref_midi, est_interval, est_midi):
    '''
    Takes a reference note track and an estimated note track, removes estimated
    notes that are not overlapped by reference notes and counts the number of
    interactions needed.

    :parameters:
        - ref_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in reference note track.
        - est_interval : np.ndarray, shape=(n_events, 2)
            Onset and offset time of each note in estimated note track.
        - est_midi : np.ndarray
            Array of MIDI note values (not necessarily integer) 
            in estimated note track.
    :returns:
        - new_interval : np.ndarray, shape=(n_events, 2)
            Intervals without the deleted notes.
        - new_midi : np.ndarray
            Array of MIDI note values of the non-deleted notes.
    '''

    n_est = len(est_midi)
    time_overlaps, timepitch_overlaps = calculate_overlaps(ref_interval,
                                                           ref_midi,
                                                           est_interval,
                                                           est_midi)
    est_overlapped = set([it[1] for it in time_overlaps])
    est_non_overlapped = [i for i in range(n_est) if not i in est_overlapped]

    edit_count_naive = len(est_non_overlapped)

    edit_count = 0
    old_note = -2
    for note in est_non_overlapped:
        if (note - old_note) > 1:
            edit_count += 1
        old_note = note

    est_overlapped = np.array(list(est_overlapped))
    new_interval = est_interval[est_overlapped,:]
    new_midi = est_midi[est_overlapped]

    return new_interval, new_midi, edit_count_naive, edit_count

def calculate_overlaps(interval_A, midi_A, interval_B, midi_B, 
                          midi_threshold=0.5, onset_threshold=0.05):
    '''
    Calculate segmentation metrics from Molina's paper and some additional
    ones.

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
        TODO
    '''
    
    n_A = len(midi_A)

    time_overlaps = collections.OrderedDict()
    timepitch_overlaps = collections.OrderedDict()

    for i, iv_A in enumerate(interval_A):
        for j, iv_B in enumerate(interval_B):
            is_overlap = iv_A[0] <= iv_B[1] and iv_A[1] >= iv_B[0]

            if is_overlap:
                is_correct_pitch = abs(midi_A[i]-midi_B[j]) < midi_threshold
                t_overlap = min(iv_B[1], iv_A[1]) - max(iv_B[0], iv_A[0])
                rel_overlap_A = t_overlap/(iv_A[1]-iv_A[0])
                rel_overlap_B = t_overlap/(iv_B[1]-iv_B[0])
                time_overlaps[(i,j)] = (t_overlap,
                                        rel_overlap_A, rel_overlap_B)

                if is_correct_pitch:
                    timepitch_overlaps[(i,j)] = (t_overlap,
                                                 rel_overlap_A, rel_overlap_B)
    return time_overlaps, timepitch_overlaps

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
                              columns=['COnPOff', 'COnP', 'COn'])
    
    conpoff_matches = []
    conp_matches = []
    con_matches = []

    for i, iv_A in enumerate(interval_A):
        offset_thresh = iv_A[0] + (iv_A[1]-iv_A[0]) * np.array([0.8, 1.2])

        for j, iv_B in enumerate(interval_B):
            is_overlap = iv_A[0] <= iv_B[1] and iv_A[1] >= iv_B[0]
            is_correct_pitch = abs(midi_A[i]-midi_B[j]) < midi_threshold
            is_matched_onset = abs(iv_A[0]-iv_B[0]) < onset_threshold
            is_matched_offset = iv_B[1] > offset_thresh[0] and iv_B[1] < offset_thresh[1]

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

    (del_interval, del_midi, 
     del_edit_count_naive, del_edit_count) =  pseudomanual_delete(ref_interval,
                                                          ref_midi,
                                                          est_interval,
                                                          est_midi)
    del_frame_times, del_frame_midi = rasterize_notes(del_interval, del_midi,
                                                      max_time)
    (scores['Del Voicing Recall'],
    scores['Del Voicing False Alarm']) = filter_kwargs(voicing_measures,
                                                       ref_frame_midi>0,
                                                       del_frame_midi>0, **kwargs)

    scores['Del Raw Pitch Accuracy'] = filter_kwargs(raw_pitch_accuracy,
                                                     ref_frame_midi>0,
                                                     ref_frame_midi*100,
                                                     del_frame_midi>0,
                                                     del_frame_midi*100,
                                                     **kwargs)
    scores['Del Raw Chroma Accuracy'] = filter_kwargs(raw_chroma_accuracy,
                                                       ref_frame_midi>0,
                                                       ref_frame_midi*100,
                                                       del_frame_midi>0,
                                                       del_frame_midi*100,
                                                       **kwargs)

    scores['Del Overall Accuracy'] = filter_kwargs(overall_accuracy,
                                                    ref_frame_midi>0,
                                                    ref_frame_midi*100,
                                                    del_frame_midi>0,
                                                    del_frame_midi*100,
                                                    **kwargs)
    

    (del_precision, 
     del_recall, 
     del_fmeasure) = calculate_match_metrics(ref_interval, ref_midi,
                                             del_interval, del_midi)
    # scores['Del Precision COnPOff'] = del_precision.COnPOff
    # scores['Del Recall COnPOff'] = del_recall.COnPOff
    scores['Del F Measure COnPOff'] = del_fmeasure.COnPOff

    # scores['Del Precision COnP'] = del_precision.COnP
    # scores['Del Recall COnP'] = del_recall.COnP
    scores['Del F Measure COnP'] = del_fmeasure.COnP

    # scores['Del Precision COn'] = del_precision.COn
    # scores['Del Recall COn'] = del_recall.COn
    scores['Del F Measure COn'] = del_fmeasure.COn

    # scores['Del Count Naive'] = del_edit_count_naive
    # scores['Del Count'] = del_edit_count
    # scores['Del Prop Naive'] = float(del_edit_count_naive)/len(ref_midi)
    # scores['Del Prop'] = float(del_edit_count)/len(ref_midi)

    (mer_interval, mer_midi) =  pseudomanual_merge(ref_interval,
                                                          ref_midi,
                                                          del_interval,
                                                          del_midi)
    mer_frame_times, mer_frame_midi = rasterize_notes(mer_interval, mer_midi,
                                                      max_time)

    (scores['Mer Voicing Recall'],
    scores['Mer Voicing False Alarm']) = filter_kwargs(voicing_measures,
                                                       ref_frame_midi>0,
                                                       mer_frame_midi>0, **kwargs)

    scores['Mer Raw Pitch Accuracy'] = filter_kwargs(raw_pitch_accuracy,
                                                     ref_frame_midi>0,
                                                     ref_frame_midi*100,
                                                     mer_frame_midi>0,
                                                     mer_frame_midi*100,
                                                     **kwargs)
    scores['Mer Raw Chroma Accuracy'] = filter_kwargs(raw_chroma_accuracy,
                                                       ref_frame_midi>0,
                                                       ref_frame_midi*100,
                                                       mer_frame_midi>0,
                                                       mer_frame_midi*100,
                                                       **kwargs)

    scores['Mer Overall Accuracy'] = filter_kwargs(overall_accuracy,
                                                    ref_frame_midi>0,
                                                    ref_frame_midi*100,
                                                    mer_frame_midi>0,
                                                    mer_frame_midi*100,
                                                    **kwargs)

    (mer_precision, 
     mer_recall, 
     mer_fmeasure) = calculate_match_metrics(ref_interval, ref_midi,
                                             mer_interval, mer_midi)
    # scores['Mer Precision COnPOff'] = mer_precision.COnPOff
    # scores['Mer Recall COnPOff'] = mer_recall.COnPOff
    scores['Mer F Measure COnPOff'] = mer_fmeasure.COnPOff

    # scores['Mer Precision COnP'] = mer_precision.COnP
    # scores['Mer Recall COnP'] = mer_recall.COnP
    scores['Mer F Measure COnP'] = mer_fmeasure.COnP

    # scores['Mer Precision COn'] = mer_precision.COn
    # scores['Mer Recall COn'] = mer_recall.COn
    scores['Mer F Measure COn'] = mer_fmeasure.COn

    scores['Mer Overall Accuracy'] = filter_kwargs(overall_accuracy,
                                                    ref_frame_midi>0,
                                                    ref_frame_midi*100,
                                                    mer_frame_midi>0,
                                                    mer_frame_midi*100,
                                                    **kwargs)

    return scores
