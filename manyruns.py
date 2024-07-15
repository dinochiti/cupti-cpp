#!/usr/bin/env python3
import os
import re
import sys
from subprocess import check_output
from time import sleep

#
#  run many times with output formatted for easy graphing
#

if len(sys.argv) != 2:
    print("{} takes a single command line argument for number times to run each variant".format(sys.argv[0]))
    exit()

ITERATIONS = int(sys.argv[1]) + 2

NUM_POINTS = [" 32 ", " 64 ", " 1K ", " 1M ", " 16M "]
BLOCK_SIZE = [" 32 ", " 64 ", " 128 ", " 256 "]

total_data = [["block size", "32", "64", "128", "256"]]

for num_points in NUM_POINTS:
    new_total_row = [num_points + " points"]
    total_data.append(new_total_row)
    for block_size in BLOCK_SIZE:
        cmd = "./bin/cc {} {}".format(num_points, block_size)
        print(cmd)
        total_time = 0.0
        total_times = []
        for iteration in range(ITERATIONS):
            out = check_output(cmd, shell=True).decode("ascii")
            l = re.search(" time (.*) ns", out)
            if l is not None:
                total_times.append(float(l.group(1)))
            else:
                print("Error! Unexpected output format\n")
                print(out, "\n")
                exit()
        total_times.remove(max(total_times))
        total_times.remove(min(total_times))
        total_time = sum(total_times)
        total_time /= len(total_times)
        new_total_row.append("{:.0f}".format(total_time))

print("\ntotal time (ms)\n")
for row in total_data:
    print("\t".join(row))
