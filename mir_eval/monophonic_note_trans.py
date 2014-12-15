# CREATED:2014-12-15 by Matthias Mauch <mail@matthiasmauch.net>
'''


'''

import numpy as np
import pandas as pd
import scipy.interpolate
import collections
import warnings
from .util import filter_kwargs

def get_note_midi(interval, times, f0s):
    temp = [f0s[i] for i, t in enumerate(times)
            if t >= interval[0] and t < interval[1]]
    midi = np.log2(np.array(temp)/ 440) * 12 + 69
    return np.median(midi)

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


def pseudomanual_merge(ref_interval, ref_midi, est_interval, est_midi,
                       times=[], f0s=[]):
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
            merge_plan.extend([el] for 
                              el in range(last_est, min(est_overlap_ind)))
            merge_plan.append(est_overlap_ind)
            last_est = max(est_overlap_ind) + 1

    merge_plan.extend([el] for el in range(last_est, n_est))

    # compose new notes from merge plan

    new_interval = list()
    new_midi = list()

    merge_count = 0

    for group in merge_plan:
        if len(group) > 1:
            merge_count += 1
        dur = [ei[1]-ei[0] for ei in est_interval[group, :]]
        onset = min([ei[0] for ei in est_interval[group, :]])
        offset = max([ei[1] for ei in est_interval[group, :]])
        if f0s != []:
            midi = get_note_midi((onset, offset), times, f0s)
        else:
            midi = est_midi[group][np.argmax(dur)]
        new_interval.append((onset, offset))
        new_midi.append(midi)

    return np.array(new_interval), np.array(new_midi), merge_count


def pseudomanual_delete(ref_interval, ref_midi, 
                        est_interval, est_midi):
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

def pseudomanual_split(ref_interval, ref_midi,
                       est_interval, est_midi,
                       times=[], f0s=[],
                       thresh=0.1):
    '''
    Takes a reference note track and an estimated note track, splits estimated
    notes where there are small gaps in the reference note track and counts 
    the number of interactions needed.

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
        - split_count : int
            Number of splits executed.
    '''

    n_est = len(est_midi)
    n_ref = len(ref_midi)
    time_overlaps, timepitch_overlaps = calculate_overlaps(ref_interval,
                                                           ref_midi,
                                                           est_interval,
                                                           est_midi)
    split_plan = dict()
    old_off = ref_interval[0, 1]

    for i_ref in range(1, n_ref):
        on = ref_interval[i_ref, 0]

        if old_off - on < thresh:
            splittable = [i for i, iv in enumerate(est_interval)
                          if iv[0] < old_off-thresh and iv[1] > on+thresh]
            if splittable:
                split_plan[splittable[0]] = on

        old_off = ref_interval[i_ref, 1]

    split_count = len(split_plan)

    new_interval = []
    new_midi = []
    for i_est in range(n_est):
        if i_est in split_plan:
            iv1 = [est_interval[i_est, 0], split_plan[i_est]]
            iv2 = [split_plan[i_est], est_interval[i_est, 1]]
            new_interval.append(iv1)
            new_interval.append(iv2)
            if f0s != []:
                new_midi.append(get_note_midi(iv1, times, f0s))
                new_midi.append(get_note_midi(iv2, times, f0s))
            else:
                new_midi.append(est_midi[i_est])
                new_midi.append(est_midi[i_est])
        else:
            new_interval.append(est_interval[i_est, :])
            new_midi.append(est_midi[i_est])

    return np.array(new_interval), np.array(new_midi), split_count