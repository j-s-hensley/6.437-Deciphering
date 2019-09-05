#! /usr/bin/python
import sys
from numpy import random

def main(inp, out):
    # Load and encrypt a piece of text.
    alphabet = list('abcdefghijklmnopqrstuvwxyz .')
    with open(inp,'r') as file:
        a = file.read()
    b = [i for i in a.lower() if i in alphabet]
    cipher = random.permutation(alphabet)
    translate = dict(zip(alphabet,cipher))
    c = ''.join([translate[i] for i in b])
    with open(out,'w') as file:
        file.write(c)


if __name__ == '__main__':
    try:
        input_file, output_file = sys.argv[1:3]
        main(input_file, output_file)
    except ValueError:
        print('Usage: python encode.py [your input file] [desired output file name]')
