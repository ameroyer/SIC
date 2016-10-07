
"""
Remove audio samples that are too short
"""

import os
import errno
import ConfigParser as cfg
import argparse
import subprocess
import shutil
import wave
from collections import defaultdict
import contextlib


def get_wav_length(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = float(frames) / float(rate)
        return duration



if __name__ == "__main__":

    # ========== Custom parameters
    parser = argparse.ArgumentParser(description='Extracting Features using HTK.')
    parser.add_argument('-i', '--input', dest = 'samples_folder', type = str, help='Folder containing the audio samples.')
    parser.add_argument('-l', '--limit', dest = 'limit', type=float, default=0.16, help='Folder containing the audio samples.')
    parser.add_argument('-ho', '--homonyms', dest = 'homonyms', type=str, help='path to homonyms file.')
    args = parser.parse_args()


    #Remove too short segments
    print '>>>>>>>>>>>>>>>>> Removing samples from %s with length smaller than %s'%(args.samples_folder, args.limit)

    wav_files = [(os.path.join(args.samples_folder, x), get_wav_length(os.path.join(args.samples_folder, x))) for x in os.listdir(args.samples_folder) if x.endswith('.wav')]
    wav_files = sorted(wav_files, key= lambda x: x[1])

    for f,l in wav_files:
        if l < args.limit:
            print '%s: %s s (Removed)' %(f,l)
            shutil.move(f, os.path.join(os.path.dirname(f), 'Discarded', os.path.basename(f)))
        else:
            print '%s: %s s' %(f,l)

    #Remove too small classes
    print '\n\n>>>>>>>>>>>>>>>>>>>> Removing classes with less than 10 occurrences'
    #Load homonys
    common_class = {}
    if args.homonyms is not None:
        with open(args.homonyms, 'r') as f:
            for line in f:
                homo = line.split()
                for h in homo:
                    common_class[h] = homo[0]

    #Browse
    class_to_samples = defaultdict(lambda:[])
    samples = [x for x in os.listdir(args.samples_folder) if x.endswith('*.wav')]
    for s in samples:
        id = samples.split('.',1)[0]
        try:
            id = common_class[h]
        except KeyError:
            pass
        class_to_samples[id].append(x)

    #Remove
    ordered_classes = sorted(class_to_samples.keys(), key= lambda x: len(class_to_samples[x]))
    for k in ordered_classes:
        class_samples = class_to_samples[k]
        if len(class_samples) < 10:
            print '%s: %s samples (Removed)' %(k, len(class_samples))
            for x in class_samples:
                shutil.move(os.path.join(args.samples_folder, x), os.path.join(args.samples_folder, 'Discarded', x))
        else:
            print '%s: %s samples' %(k, len(class_samples))
