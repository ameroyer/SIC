#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**error.py.** error module
"""

__author__ = "Amélie Royer"
__email__ = "amelie.royer@ens-rennes.fr"
__date__ = "2015"

import sys

def warning(obj):
    """
    Print a warning on the error stream.

    Args:
     * ``obj`` (*str*): warning message
    """
    print >> sys.stderr, ('WARNING: ' + obj)
    sys.stderr.flush()


class InputError(Exception):
    """
    Exception raised  when an error is found in the input data.
    """
    pass


class ConfigError(Exception):
    """
    Exception raised  when a compatibility error is found in the configuration options.
    """
    pass


class ParsingError(Exception):
    """
    Exception raised  when a parsing error is found the configuration options.
    """
    pass


def signal_handler(signal, frame):
    """
    Handles the Keyboard interrupt signal in case of multi-process execution.
    """
    print 'Process interrupted by Keyboard signal'
    sys.exit(0)
