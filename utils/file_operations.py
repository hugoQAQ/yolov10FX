import pickle
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        pkl_file = pickle.load(f)
    return pkl_file
def save_pkl(file_path, pkl_file):
    with open(file_path, 'wb') as f:
        pickle.dump(pkl_file, f)