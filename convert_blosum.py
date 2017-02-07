#!/usr/bin/env python

""" Convert BLOSUM62 from log odds to a probability matrix

This script is used to generate ``hw2/blosum62.py`` by calling::

    python convert_blosum.py > hw2/blosum62.py

BLOSUM62 uses the log odds of the probability of an amino acid substitution
to align protien sequences. The CPASS similarity metric I'm using wants the
normalized substitution probabilities, so this script converts the scores
backwards into probabilities.

Henikoff, S., and Henikoff, J.G. (1992). Amino acid substitution matrices from
protein blocks. Proceedings of the National Academy of Sciences 89,
10915–10919.

BLOSUM62 matrix at `data/blosum62.txt` from
`NIH BLAST <ftp://ftp.ncbi.nih.gov/blast/matrices/BLOSUM62>`_
"""

import pathlib

BLOSUM62_FILE = pathlib.Path('data/blosum62.txt')
AA_THREE_LETTERS = {
    'A': 'ala',
    'R': 'arg',
    'N': 'asn',
    'D': 'asp',
    'B': 'asx',
    'C': 'cys',
    'E': 'glu',
    'Q': 'gln',
    'G': 'gly',
    'H': 'his',
    'I': 'ile',
    'L': 'leu',
    'K': 'lys',
    'M': 'met',
    'F': 'phe',
    'P': 'pro',
    'S': 'ser',
    'T': 'thr',
    'W': 'trp',
    'Y': 'tyr',
    'V': 'val',
    'X': '*',  # Extra codes, unused by this algorithm
    'Z': '*',
    '*': '*',
}

blosum62 = {}

with BLOSUM62_FILE.open('rt') as fp:

    # Skip through the commented out header
    for line in fp:
        if line.strip().startswith('#'):
            continue
        break

    header = line.split(' ')
    header = [h.strip() for h in header]
    header = [h for h in header if h != '']
    header = [AA_THREE_LETTERS[h].upper() for h in header]

    for line in fp:
        line = [l.strip() for l in line.split(' ')]
        line = [l for l in line if l != '']

        # Convert scores to int
        key, *scores = line
        key = AA_THREE_LETTERS[key].upper()
        if key == '*':
            continue

        # BLOSUM62 log odds is 2 * log_base2(p_ij / (q_i * q_j))
        scores = [2**(float(s)/2.0) for s in scores]

        assert len(scores) == len(header)
        for h, s in zip(header, scores):
            if h == '*':
                continue

            blosum62[(h, key)] = s

# Sanity check
# Assert that the matrix is symmetric
differences = []
for aa1, aa2 in blosum62:
    assert blosum62[aa1, aa2] == blosum62[aa2, aa1]

print('BLOSUM62 = {')
for aa1, aa2 in sorted(blosum62):
    print('    ("{}", "{}"): {},'.format(aa1, aa2, blosum62[aa1, aa2]))
print('}')
