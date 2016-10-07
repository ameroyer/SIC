#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

"""
**basic_hmm.py.** Generates a basic initial HMM for HTK training.
"""

import os

def generate_basic_hmm(features_type, components, name, output_folder, n_state = 10):
    """
    On-the-fly generation of an initial HMM

    Args:
     * ``feature_type`` (*str*): HTK features target kind.
     * ``components`` (*int*): number of base components.
     * ``name`` (*str*) : HMM/model name.
     * ``output_folder`` (*str*): folder where to output the HMM.
     * ``n_state`` (*int, optional*): number of states in the HMM. Defaults to 10.
    """
    #Header
    header = """~o <VecSize> %d <%s>
~h "%s"
<BeginHMM>
<NumStates> %d"""%(components, features_type, name, n_state + 2)

    #States
    states = [0] * n_state
    for n in xrange(n_state):
        states[n] = """<State> %d
<Mean> %d
%s
<Variance> %d
%s""" %(n + 2, components, ' '.join(['0.0']*components), components, ' '.join(['1.1']*components))
    states = '\n'.join(states)

    #Transition matrix
    lines = [0] * (n_state + 2)
    lines[0] = '0.0 1.0 %s'%(' '.join(['0.0']*n_state))
    lines[-1] = ' '.join(['0.0']*(n_state+2))

    #Left/Right transition matrix
    for n in xrange(n_state):
        lines[n+1] = '%s 0.5 0.5 %s'%(' '.join(['0.0']*(n + 1)), ' '.join(['0.0']*(n_state - n - 1)))
    transmat = "<TransP> %d\n%s"%(n_state + 2, '\n'.join(lines))

    #Final
    basichmm = "%s\n%s\n%s\n<EndHMM>"%(header, states, transmat)
    output_path = os.path.join(output_folder, name)
    with open(output_path, 'w') as f:
        f.write(basichmm)

    return output_path
