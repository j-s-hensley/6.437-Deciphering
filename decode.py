#! /usr/bin/python

import numpy as np
import numpy.random
import sys
from scipy.io import loadmat

def preprocess():
    """
    Load model parameters from external files.
    Inputs :
        language_parameters.mat (file) : contains model parameters
    Returns :
        let_trans (np.array) : matrix where let_trans[i,j] is probability of
                               transitioning from character j to character i,
                               given the previous character was j
        let_prob (np.array) : the marginal probability of each character
        alphabet (str) : a string containing the 26 letters, space, and period,
                         with the order corresponding to the probabilities above
    """
    np.seterr(divide='ignore')
    matlab_parameters = loadmat('language_parameters.mat')
    let_trans = np.log(matlab_parameters['letter_transition_matrix'])
    let_trans[np.isnan(let_trans)] = -np.inf
    let_prob = np.log(matlab_parameters['letter_probabilities'])
    alphabet = 'abcdefghijklmnopqrstuvwxyz .'
    return let_trans, let_prob, alphabet


def decode(ciphertext,filename):
    """
    Take a piece of text that has been encrypted with a simple substitution
    cipher and decrypt it using the Metropolis-Hastings algorithm.
    Inputs :
        ciphertext (str) : the encrypted text. Note that we assume (1) all
                           characters are lowercase (2) the only characters used
                           are the 26 letters of the English alphabet, period,
                           and space (3) a period is always followed by a space.
        filename (str) : a file to save the plaintext as
        save (bool) : whether or not to save the plaintext as a new text file
    Returns :
        deciphered (str) : the decrypted text
    """
    let_trans, let_prob, alphabet = preprocess()
    x = [alphabet.index(letter) for letter in ciphertext]
    f = initf(x,alphabet)

    counter = 0
    threshold = 10**8/max(len(ciphertext),10**3)
    while counter < threshold:
        fprime = genf(f)
        a = min(1,compare_log_probabilites(fprime,f,x,let_trans,let_prob))
        if np.random.binomial(1,a):
            counter = 0
            f = fprime
        else:
            counter += 1

    deciphered = ''.join([alphabet[f[i]] for i in x])
    with open(filename,'w') as file:
        file.write(deciphered)
    return deciphered


def initf(x,alphabet):
    """
    Initialize a possible cipher.
    Inputs :
        x (list of ints) : the input ciphertext, rendered as the integers
                           corresponding to each character
        alphabet (str) : a string containing the 26 letters, space, and period,
                         with the order corresponding to the probabilities above
    Returns :
        f (list of ints) : a possible cipher, where e.g. a '27' in the first
                           position indicates that a '.' in the ciphertext
                           should be replaced with 'a'
    """
    candidate_pairs = []

    for i in range(len(alphabet)):
        pos = [j for j in range(len(x)-1) if x[j]==i]
        followers = [x[j+1] for j in pos]
        if len(followers) > 0 and all([fol==followers[0] for fol in followers]):
            candidate_pairs += [(i,followers[0])]

    i,j = np.random.choice(range(len(candidate_pairs)),size=2,replace=False)
    period,space = candidate_pairs[i]
    q,u = candidate_pairs[j]
    f = list(np.random.permutation(range(len(alphabet))))

    f.remove(alphabet.index('q'))
    f.remove(alphabet.index('u'))
    f.remove(alphabet.index(' '))
    f.remove(alphabet.index('.'))

    for ind,let in sorted([(q,'q'),(u,'u'),(space,' '),(period,'.')],key=lambda y: y[0]):
        f.insert(ind,alphabet.index(let))
    return f


def genf(f):
    """
    Generate a possible cipher by swapping the position of two elements.
    Inputs :
        f (list of ints) : a possible cipher, where e.g. a '27' in the first
                           position indicates that a '.' in the ciphertext
                           should be replaced with 'a'
    Returns:
        fprime (list of ints) : a possible cipher that is identical to f in all
                                but two elements
    """
    fprime = f[:]
    a,b = np.random.choice(range(28),size=2,replace=False)
    fprime[a], fprime[b] = fprime[b], fprime[a]
    return fprime


def compare_log_probabilites(fprime,f,x,let_trans,let_prob):
    """
    Compare the log probabilities of two possible decipherings of the ciphertext.
    Inputs :
        fprime (list of ints) : a proposed cipher to replace the current one
        f (list of ints) : the current cipher
        x (list of ints) : the input ciphertext, rendered as the integers
                           corresponding to each character
        let_trans (np.array) : matrix where let_trans[i,j] is probability of
                               transitioning from character j to character i,
                               given the previous character was j
        let_prob (np.array) : the marginal probability of each character
    Returns :
        p (float) : the probability of the proposed cipher producing correct
                    plaintext divided by the probability of the current cipher
                    producing correct plaintext, used in the MH algorithm
    """
    fpx = [fprime[i] for i in x]
    fx = [f[i] for i in x]
    a = let_prob[fpx[0]] + sum([let_trans[fpx[i],fpx[i-1]] for i in range(1,len(x))])
    b = let_prob[fx[0]] + sum([let_trans[fx[i],fx[i-1]] for i in range(1,len(x))])
    if a == b:
        p = 1
    else:
        p = np.exp(a-b)
    return p


if __name__ == '__main__':
    try:
        input_file, output_file = sys.argv[1:3]
        with open(input_file,'r') as file:
            input_text = file.read()
        decode(input_text, output_file)
    except ValueError:
        print('Usage: python decode.py [your input file] [desired output file name]')
