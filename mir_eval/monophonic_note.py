# CREATED:2014-12-08 by Matthias Mauch <mail@matthiasmauch.net>
'''


'''

import numpy as np
import scipy.interpolate
import collections
import warnings
from . import util


def evaluate(ref_interval, ref_midi, est_interval, est_midi, **kwargs):
    '''
    Evaluate two note transcriptions, where the first is treated as the
    reference (ground truth) and the second as the estimate to be evaluated
    (prediction).

    :usage: TODO
        >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
        >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
        >>> scores = mir_eval.melody.evaluate(ref_time, ref_freq,
                                              est_time, est_freq)
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

    :raises:
        - ValueError
            Thrown when the provided annotations are not valid.
    '''

    # Compute metrics
    scores = collections.OrderedDict()

    (scores['Voicing Recall'],
     scores['Voicing False Alarm']) = util.filter_kwargs(voicing_measures,
                                                         ref_voicing,
                                                         est_voicing, **kwargs)

    scores['Raw Pitch Accuracy'] = util.filter_kwargs(raw_pitch_accuracy,
                                                      ref_voicing, ref_cent,
                                                      est_voicing, est_cent,
                                                      **kwargs)

    return scores
