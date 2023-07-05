#!/usr/bin/env python

import sys
import os

def patch_files(file1, file2):
    with open(file2, 'r') as fh:
        patchlist = fh.readlines()
    count = 0

    with open(file1, 'r+b') as fh:
        for line in patchlist:
            count += 1
            offset = int(line[0:8], 16)
            value = bytes.fromhex(line[10:12])
            print(offset, value)
            fh.seek(offset)
            fh.write(value)

if __name__ == '__main__':

    f1 = sys.argv[1]
    f2 = sys.argv[2]

    patch_files(f1, f2)

