import numpy as np
import os
import pandas as pd
import pickle

def ark_parser(ark_path, filename):
    df = pd.read_csv(os.path.join(ark_path, filename), header=None, delimiter=' ')
    name = ''
    input = []
    inputlist = []
    namelist = []

    for _, i in df.iterrows():
        curr_name = '_'.join(i.values[0].split('_')[:2])
        if(name != curr_name):
            
            if(name != ''):
                inputlist.append(np.array(input, dtype=np.float32))
                namelist.append(name)
                input = []

            name = curr_name

        input.append(i.values[1:])

    if(name != ''):
        inputlist.append(np.array(input, dtype=np.float32))
        namelist.append(name)
    print('done')
    return inputlist, namelist

def lab_parser(lab_path):

    df = pd.read_csv(os.path.join(lab_path, 'train.lab'), header=None, delimiter=',')
    name = ''
    labdict = {}
    input = []
    for _, i in df.iterrows():
        curr_name = '_'.join(i.values[0].split('_')[:2])
        if(name != curr_name):
            if(name != ''):
                labdict[name] = input
                input = []
            name = curr_name
        input.append(i.values[1])
    if(name != ''):
        labdict[name] = input
        
    return labdict

def convert_testing_data(mfccPath):
    """convert testing data to pkl file"""
    inputlist, inputnamelist = ark_parser(mfccPath, 'test.ark')

    print("%d sample in testing set" % len(inputlist))
    with open('../data/test_data.pkl', 'wb') as test_data:
        pickle.dump(inputlist, test_data)
    with open('../data/test_name.pkl', 'wb') as test_name:
        pickle.dump(inputnamelist, test_name)


def convert_data(mfccPath, labelPath):
    """convert training data to pkl file"""
    inputlist, inputnamelist = ark_parser(mfccPath, 'train.ark')
    labeldict = lab_parser(labelPath)

    label = []
    assert len(inputnamelist) == len(labeldict.keys())

    for name in inputnamelist:
        label.append(labeldict[name])

    with open('../data/train_data.pkl', 'wb') as train_data:
        pickle.dump(inputlist, train_data)

    with open('../data/train_label.pkl', 'wb') as train_label:
        pickle.dump( label, train_label )

def phone_map_reader(path_to_phone_map):
    """48 phone to 39 phone"""
    mapping = dict()
    phn = []
    group_phn = []
    with open(path_to_phone_map) as f:
       
        for line in f:
            m = line.strip().split('\t')
            phn.append(m[0])
            if m[1] not in group_phn:
                group_phn.append(m[1])
            mapping[m[0]] = m[1]

    return phn, group_phn, mapping

def phone_char_reader(path_to_phone_char_map):
    """"map 48 phone to 26 character"""
    mapping = dict()
    with open(path_to_phone_char_map) as f:
        for line in f:
            m = line.strip().split('\t')
            mapping[m[0]] = m[2]

    return mapping 

def phone_int_mapping(path_to_phone_char_map):
    """"map 48 phone to 26 character"""
    mapping = dict()
    with open(path_to_phone_char_map) as f:
        for line in f:
            m = line.strip().split('\t')
            mapping[m[0]] = int(m[1])

    return mapping 



def main():
    mfcc_path = '../data/mfcc'
    label_path = '../data/label'
    #convert_data(mfcc_path, label_path)
    convert_testing_data(mfcc_path)


if __name__ == "__main__":
    main()