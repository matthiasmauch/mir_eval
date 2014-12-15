#!/usr/bin/env python
'''
CREATED:2014-12-15 by Matthias Mauch <mail@matthiasmauch.net>
(adapted from melody_eval.py)

Transform estimated monophonic note tracks with three consecutive pseudomanual
edit actions, and save to new files.

Usage:

./pseudomanual_edits.py TRUTH.TXT PREDICTION.TXT
(CSV files also accepted)

'''

import argparse
import sys
import os

import mir_eval

from mir_eval.monophonic_note_trans import *

def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval monophonic note '
                                                 'extraction evaluation')

    parser.add_argument('-p',
                        dest='f0_file',
                        default=None,
                        type=str,
                        action='store',
                        help='path to the optional time/f0 file')

    parser.add_argument('reference_file',
                        action='store',
                        help='path to the ground truth annotation')

    parser.add_argument('estimated_file',
                        action='store',
                        help='path to the estimation file')

    return vars(parser.parse_args(sys.argv[1:]))

def write_note_txt(interval, midi, fn=None):
    if fn == None:
        f = sys.stdout
    else:
        f = open(fn, 'w')
    for i, iv in enumerate(interval):
        print >>f, "{0}\t{1}\t{2}".format(iv[0], iv[1], midi[i])

if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Load in the data from the provided files
    (ref_interval,
     ref_midi) = mir_eval.io.load_value_intervals(parameters['reference_file'])
    (est_interval,
     est_midi) = mir_eval.io.load_value_intervals(parameters['estimated_file'])
    if parameters['f0_file']:
        (times, f0s) = mir_eval.io.load_time_series(parameters['f0_file'], 
                                                    delimiter=",")
    else:
        times = None
        f0s = None


    est_main, est_ext = os.path.splitext(parameters['estimated_file'])

    (spl_interval, spl_midi,
                   split_count) = pseudomanual_split(ref_interval, ref_midi,
                                                     est_interval, est_midi,
                                                     times, f0s)

    print parameters['f0_file']

    print >>sys.stderr, "{0} splits".format(split_count)
    spl_fn = "{0}_spl{1}".format(est_main, est_ext)
    write_note_txt(spl_interval, spl_midi, fn=spl_fn)

    (mer_interval, mer_midi,
                   merge_count) = pseudomanual_merge(ref_interval, ref_midi,
                                                     spl_interval, spl_midi,
                                                     times, f0s)
    print >>sys.stderr, "{0} merges".format(merge_count)
    mer_fn = "{0}_mer{1}".format(est_main, est_ext)
    write_note_txt(mer_interval, mer_midi, fn=mer_fn)

    (del_interval, del_midi,
     del_naive_count, del_count) = pseudomanual_delete(ref_interval, ref_midi,
                                                       mer_interval, mer_midi)
    print >>sys.stderr, "{0} deletes".format(del_count)
    del_fn = "{0}_del{1}".format(est_main, est_ext)
    write_note_txt(del_interval, del_midi, fn=del_fn)