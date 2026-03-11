import pickle

def save_object(file_path, obj):

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)