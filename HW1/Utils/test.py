import pickle

f=open('data/train_label.pkl', 'rb')
a = pickle.load(f)

"""check the training data if there are phone with aba or abc pattern"""
for i in a:
    for j in range(1, len(i)-1):
        if(i[j-1] == i[j+1] and i[j-1] != i[j]):
            print(i[j-10:j+10])
            break
        if(i[j-1] != i[j+1] and i[j] != i[j-1] and i[j] != i[j+1]):
            print(i[j-10:j+10])
            break
