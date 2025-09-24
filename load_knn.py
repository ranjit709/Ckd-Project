import pickle
import numpy as np

#Loading model
with open('../SavedModels/knn.pickle','rb') as f:
    model=pickle.load(f)

#Prediction

#Prediction
k=np.array([[1.017408,1,0,0,99,11.7,48,5000,2.5,0,1,1,1,0]])
predict_dt=model.predict(k)
predict_dt=int(predict_dt.item())
# sclr=np.squeeze(predict_dt)
classes=np.array(['Normal','Kidney disease detected'])
predict_dt
print(classes[predict_dt])