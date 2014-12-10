#!/usr/bin/env python
'''
CREATED:2014-12-08 by Matthias Mauch <mail@matthiasmauch.net>
(adapted from melody_eval.py)

Compute metrics for the evaluation of monophonic note sequences

Usage:

./monophonic_note_eval.py TRUTH.TXT PREDICTION.TXT
(CSV files also accepted)

For a detailed explanation of the measures please refer to:

@article{molina2014evaluation,
  title={Evaluation framework for automatic singing transcription},
  author={Molina, E. and Barbancho, A. M. and Tard{\'o}n, L. J. and Barbancho, I.},
  booktitle={Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011)},
  pages={567--572},
  year={2014}
}

'''

import argparse
import sys
import os
import eval_utilities

import mir_eval


def process_arguments():
    '''Argparse function to get the program parameters'''

    parser = argparse.ArgumentParser(description='mir_eval monophonic note '
                                                 'extraction evaluation')

    parser.add_argument('-o',
                        dest='output_file',
                        default=None,
                        type=str,
                        action='store',
                        help='Store results in json format')

    parser.add_argument('reference_file',
                        action='store',
                        help='path to the ground truth annotation')

    parser.add_argument('estimated_file',
                        action='store',
                        help='path to the estimation file')

    return vars(parser.parse_args(sys.argv[1:]))


if __name__ == '__main__':
    # Get the parameters
    parameters = process_arguments()

    # Load in the data from the provided files
    (ref_interval,
     ref_midi) = mir_eval.io.load_value_intervals(parameters['reference_file'])
    (est_interval,
     est_midi) = mir_eval.io.load_value_intervals(parameters['estimated_file'])

    # Compute all the scores
    scores = mir_eval.monophonic_note.evaluate(ref_interval, ref_midi, 
                                               est_interval, est_midi)
    print "{} vs. {}".format(os.path.basename(parameters['reference_file']),
                             os.path.basename(parameters['estimated_file']))
    eval_utilities.print_evaluation(scores)

    if parameters['output_file']:
        print 'Saving results to: ', parameters['output_file']
        eval_utilities.save_results(scores, parameters['output_file'])
