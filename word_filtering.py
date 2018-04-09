#!/usr/bin/env python

import sys

r = set()
for line in sys.stdin:
    [r.add(word) for word in line.split()]

for word in r:
    print(word.lower())