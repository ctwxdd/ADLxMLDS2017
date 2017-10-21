import numpy as _np
import pdb

def normalize_mfcc(mfcc):
    """Normalize mfcc data using the following formula:
    
    normalized = (mfcc - mean)/standard deviation
    
    Args:
        mfcc (numpy.ndarray):
            An ndarray containing mfcc data.
            Its shape is [sentence_length, coefficients]
    
    Returnso
        numpy.ndarray:
            An ndarray containing normalized mfcc data with the same shape as
            the input.
    """
    
    means = _np.mean(mfcc, 0)
    stds = _np.std(mfcc, 0)
    return (mfcc - means)/stds
    
def to_one_hot(labels, num_classes=48):
    """Convert class labels from scalars to one-hot vectors."""
    
    num_labels = len(labels)
    index_offset = _np.arange(num_labels) * num_classes
    labels_one_hot = _np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels] = 1
    return labels_one_hot
    
    
    
def _enumerate_context(i, sentence, num_frames):
    r = range(i-num_frames, i+num_frames+1)
    r = [x if x>=0 else 0 for x in r]
    r = [x if x<len(sentence) else len(sentence)-1 for x in r]
    return sentence[r]
