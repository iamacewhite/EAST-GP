import numpy as np

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def char_to_int(char):
    temp = ord(char)
    temp = temp - 32
    if temp > 94 or temp < 0:
        temp = 0

    return temp

def encode_str(label):
    encode_label = [char_to_int(char) for char in label]
    length = len(label)
    return encode_label,length



# train_targets = sparse_tuple_from([targets])
