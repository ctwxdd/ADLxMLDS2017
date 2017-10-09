def phone_map_reader(path_to_phone_map):
    """48 phone to 39 phone"""
    mapping = dict()
    with open(path_to_phone_map) as f:

        for line in f:
            m = line.strip().split('\t')
            mapping[m[0]] = m[1]

    return mapping 

def phone_char_reader(path_to_phone_char_map):
    """"map 48 phone to 26 character"""
    mapping = dict()
    with open(path_to_phone_char_map) as f:
        for line in f:
            m = line.strip().split('\t')
            mapping[m[0]] = m[2]

    return mapping 

        