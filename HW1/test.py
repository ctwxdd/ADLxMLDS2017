from Utils.map_reader import phone_map_reader, phone_char_reader
from Utils.load_data import load_batched_data
import tensorflow

path_to_phone_map = './data/phones/48_39.map'
path_to_phone_char_map = './data/48phone_char.map'

path_to_data_dir = './data/'

print(phone_char_reader(path_to_phone_char_map))
