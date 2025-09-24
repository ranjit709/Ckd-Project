import pickle
import numpy as np

#Loading model
with open('../SavedModels/Dt.pickle', 'rb') as f:
    model=pickle.load(f)

#Prediction

#Prediction
k=np.array([[1.025,0,0,0,0,11.7,48,12000,2.5,1,1,1,1,1]])
# k=np.array([[1,0,0,0,0,11,40,12000,7,0,0,0,0,0]])


predict_dt=model.predict(k)
predict_dt=int(predict_dt.item())
# sclr=np.squeeze(predict_dt)
classes=np.array(['Normal','Kidney disease detected'])
predict_dt
print(predict_dt)
print(classes[predict_dt])




""" 
specific_gravity', 'red_blood_cells', 'pus_cell_clumps', 'bacteria',
       'blood_glucose_random', 'haemoglobin', 'packed_cell_volume',
       'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
       'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
       'pedal_edema'
"""
# k = np.array([[SpecificGravity, RedBlood, CellClumps, Bacteria, BloodGlucose, Haemoglobin, PackedCell, WhiteBlood,
#                RedBloodCount, Hypertension, Mellitus, CoronaryArtery, Appetite, PedalEdema]])
