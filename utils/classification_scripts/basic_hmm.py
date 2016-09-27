#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**basic_hmm.py.** Generates a basic initial HMM for HTK training.
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"


def generate_basic_hmm(features_type, components, name, output_folder, n_state=12, hmm_type=1):
    """
    On-the-fly generation of an initial HMM.

    Args:
     * ``feature_type`` (*str*): HTK features target kind.
     * ``components`` (*int*): number of base components.
     * ``name`` (*str*) : HMM/model name.
     * ``output_folder`` (*str*): folder where to output the HMM.
     * ``n_state`` (*int, optional*): number of emitting states in the HMM (not counting initial and final states). Defaults to 12.
     * ``hmm_type`` (*int, optional*): determines the HMM topology to use (1: basic left/right; 2: left/right1/right2). Defaults to 1
    """
    import os
    print 'HMM Type %d, %d states'%(hmm_type, n_state)


    # Header
    header = """~o <VecSize> %d <%s>
~h "%s"
<BeginHMM>
<NumStates> %d""" % (components, features_type, name, n_state + 2)


    # States
    states = [0] * n_state
    for n in xrange(n_state):
        states[n] = """<State> %d
<Mean> %d
%s
<Variance> %d
%s""" % (n + 2, components, ' '.join('0.0' for _ in xrange(components)), components, ' '.join('1.0' for _ in xrange(components)))
    states = '\n'.join(states)

    # Transition matrix
    lines = [0] * (n_state + 2)
    lines[-1] = ' '.join('0.0' for _ in xrange(n_state+2))

    # Left/Right transition matrix
    if hmm_type == 1:
        lines[0] = '0.0 1.0 %s'%(' '.join('0.0' for _ in xrange(n_state)))
        for n in xrange(n_state):
            lines[n+1] = '%s 0.5 0.5 %s'%(' '.join('0.0' for _ in xrange(n + 1)), ' '.join('0.0' for _ in xrange(n_state - n - 1)))

    # Left/Right1/Right2 transition matrix
    elif hmm_type == 2:
        lines[0] = '0.0 0.5 0.5 %s'%(' '.join('0.0' for _ in xrange(n_state - 1)))
        lines[-2] = '%s 0.5 0.5'%(' '.join('0.0' for _ in xrange(n_state)))
        for n in xrange(n_state - 1):
            lines[n+1] = '%s 0.5 0.25 0.25 %s' % (' '.join(('0.0' for _ in xrange(n + 1))), ' '.join('0.0' for _ in xrange(n_state - n - 2)))

    transmat = "<TransP> %d\n%s" % (n_state + 2, '\n'.join(lines))

    # Final
    basichmm = "%s\n%s\n%s\n<EndHMM>" % (header, states, transmat)
    output_path = os.path.join(output_folder, name)
    with open(output_path, 'w') as f:
        f.write(basichmm)
    return output_path
