import os
import subprocess


def squeeze_generic(a, axes_to_keep):
    out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
    return a.reshape(out_s)


def current_memory_usage():
    '''Returns current memory usage (in MB) of a current process'''
    out = subprocess.Popen(['ps', '-p', str(os.getpid()), '-o', 'rss'],
                           stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    mem = float(out[1].strip()) / 1024
    return mem
