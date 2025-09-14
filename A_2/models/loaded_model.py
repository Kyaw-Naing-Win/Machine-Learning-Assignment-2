import pickle

filename = '/root/source_code/models/assignment1.model'

loaded_model = pickle.load(open(filename, 'rb'))

model = loaded_model['model']
scaler = loaded_model['scaler']
brand_le = loaded_model['brand_label']
max_power = loaded_model['max_power']
year =loaded_model[2017]
mileage =loaded_model['mileage']
