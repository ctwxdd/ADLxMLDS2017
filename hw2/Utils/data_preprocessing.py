import json
import sys
import os

training_label = 'training_label.json'
testing_label = 'testing_label.json'
def convert_to_csv(base_path):

    f = open(os.path.join(base_path, training_label))
    j = json.load(f)

    outfile = open('./Utils/train_label.csv', 'w')
    outfile.write('VideoID\tDescription\n')

    for video in j:
        for cap in video['caption']:
            try:
                outfile.write('%s\t%s\n' % (video['id'], cap))
            except:
                continue

    f = open(os.path.join(base_path, testing_label))
    j = json.load(f)

    outfile = open('./Utils/test_label.csv', 'w')
    outfile.write('VideoID\tDescription\n')

    for video in j:
        for cap in video['caption']:
            try:
                outfile.write('%s\t%s\n' % (video['id'], cap))
            except:
                continue

if __name__ == "__main__":
    print('Data Preprocessing...')
    convert_to_csv( sys.argv[1])
    print('Done')