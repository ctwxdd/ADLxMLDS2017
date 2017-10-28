import numpy as np
import os
import pandas as pd
import pickle
import preprocessing as pp
import sys


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
    with open('./test_data.pkl', 'wb') as test_data:
        pickle.dump(inputlist, test_data)
        
    with open('./test_name.pkl', 'wb') as test_name:
        pickle.dump(inputnamelist, test_name)

def convert_all_test_data(mfccPath, fbankPath, datadir):

    """convert training data to pkl file"""
    inputmfcc, inputnamemfcc = ark_parser(mfccPath, 'test.ark')
    inputfbank, inputnamefbank = ark_parser(fbankPath, 'test.ark')

    label = []
    inputlist = []
    assert len(inputnamemfcc) == len(inputnamefbank)

    for fb, mfcc in zip(inputfbank, inputmfcc):
        fb = pp.normalize_mfcc(fb)
        mfcc = pp.normalize_mfcc(mfcc)
        inputlist.append(np.concatenate((fb, mfcc), axis=1))

    with open('./test_data.pkl', 'wb') as test_data:
        pickle.dump(inputlist, test_data)

    with open('./test_name.pkl', 'wb') as test_name:
        pickle.dump(inputnamefbank, test_name)

def convert_data(DataPath, labeldict):
    """convert training data to pkl file"""

    inputlist, inputnamelist = ark_parser(DataPath, 'train.ark')
    
    label = []
    assert len(inputnamelist) == len(labeldict.keys())

    for name in inputnamelist:
        label.append(labeldict[name])

    convert_label_to_int(DataPath, '/48phone_char.map', label)

    with open('./train_data.pkl', 'wb') as train_data:
        pickle.dump(inputlist, train_data)


def convert_all_data(mfccPath, fbankPath, labeldict, datadir):
    """convert training data to pkl file"""

    inputmfcc, inputnamemfcc = ark_parser(mfccPath, 'train.ark')
    inputfbank, inputnamefbank = ark_parser(fbankPath, 'train.ark')

    label = []
    inputlist = []
    assert len(inputnamemfcc) == len(labeldict.keys()) and len(inputnamefbank) == len(labeldict.keys())

    for fb, mfcc in zip(inputfbank, inputmfcc):
        fb = pp.normalize_mfcc(fb)
        mfcc = pp.normalize_mfcc(mfcc)
        inputlist.append(np.concatenate((fb, mfcc), axis=1))

    for name in inputnamemfcc:
        label.append(labeldict[name])

    with open('./train_data.pkl', 'wb') as train_data:
        pickle.dump(inputlist, train_data)

    convert_label_to_int(datadir, datadir + '48phone_char.map', label)

def convert_label_to_int(DataPath, path_to_phone_char_map, label):

    mapping = phone_int_mapping(path_to_phone_char_map)

    labellist = []
    input = []

    for i in label:
        input = []
        for l in i:
            input.append(mapping[l])
        labellist.append(input)
        
    with open('/train_label.pkl', 'wb') as train_label:
        pickle.dump( labellist, train_label) 


def phone_int_mapping(path_to_phone_char_map):
    """"map 48 phone to 26 character"""
    mapping = dict()
    with open(path_to_phone_char_map) as f:
        for line in f:
            m = line.strip().split('\t')
            mapping[m[0]] = int(m[1])

    return mapping

def main():

    print('Loading Data mfcc + fbank...')
    datadir = sys.argv[1]
    label_path =  datadir  + 'label'
    path_to_phone_char_map = datadir + '48phone_char.map'
    mfcc_path =  datadir + 'mfcc'
    fbank_path = datadir + 'fbank'


    labeldict = lab_parser(label_path)
    
    # convert_data(mfcc_path, labeldict)
    # convert_data(fbank_path, labeldict)

    # convert_testing_data(mfcc_path)
    # convert_testing_data(fbank_path)

    convert_all_data(mfcc_path, fbank_path, labeldict, datadir)
    convert_all_test_data(mfcc_path, fbank_path, datadir)

    print('Done Loading Data mfcc + fbank...')


if __name__ == "__main__":
    main()