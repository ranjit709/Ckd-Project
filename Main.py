
from flask import Flask,render_template,request,session,send_from_directory,Response,redirect
import os
import pymysql
import pickle
import numpy as np
import pandas as pd

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app=Flask(__name__)
app.secret_key = 'aabbbccdddd'
conn = pymysql.connect(host="localhost",user="root",password="root",db="Kidney")
cursor = conn.cursor()
APP_ROOT =os.path.dirname(os.path.abspath(__file__))

#home page
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/AdminHome')
def AdminHome():
    return render_template('AdminHome.html')

@app.route('/AdminLog')
def AdminLog():
    return render_template('AdminLogin.html')
@app.route('/userlog')
def User():
    return render_template('UserLogin.html')
@app.route('/UserHome')
def UserHome():
    return render_template('UserHome.html')

@app.route('/UserRegistration')
def UserRegistration():
    return render_template('UserRegistration.html')

@app.route('/Admin1',methods=['post'])
def  Admin1():
    username = request.form.get("username")
    password = request.form.get("password")
    session['role'] = 'Admin'
    if username=='admin' and password=='admin':
        return render_template("AdminHome.html")
    else:
        return render_template("mmsg.html",msg='Invalid Login Details',color = 'bg-danger')

@app.route('/Alogout')
def alogout():
    session.pop('role',None)
    return  render_template('index.html')


@app.route('/UserRegister1',methods=['post'])
def UserRegister1():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        result = cursor.execute("select * from userreg where name='"+name+"' and email='"+email+"'")
        conn.commit()
        if result > 0:
            return render_template('mmsg.html', msg='User Already Exsit', color='bg-danger')
        else:
            result = cursor.execute("insert into userreg(name,email,phone,password)values('" + name + "','" + email + "','" + phone + "','" + password + "')")
            conn.commit()
            return render_template('mmsg.html', msg='User Registeraion success', color='bg-success')
    except Exception as e:
        return render_template('mmsg.html', msg=str(e), color='bg-danger')

@app.route('/UserLogin1',methods=['post'])
def UserLogin1():
    try:
        phone = request.form.get('phone')
        password = request.form.get('password')
        result = cursor.execute("select * from userreg where phone='"+phone+"' and password='"+password+"'")
        UserDetails = cursor.fetchall()
        print(UserDetails)
        conn.commit()
        if result > 0:
            for user in UserDetails:
                User_id = user[0]
                name = user[1]
                email = user[2]
                phone = user[3]
                session['User_id'] = User_id
                session['name'] = name
                session['phone'] = phone
                session['email'] = email
            return render_template('UserHome.html')
        else:
            return render_template('mmsg.html', msg='Invalid Login Details', color='bg-danger')
    except Exception as e:
        return render_template('mmsg.html', msg=str(e), color='bg-danger')

@app.route('/UserLogout')
def UserLogout():
    session.pop('user_id', None)
    session.pop('email', None)
    session.pop('password', None)
    return  render_template('index.html')
@app.route('/UploadData')
def UploadData():
    return render_template('UploadData.html')

@app.route('/upload1',methods=['POST'])
def upload1():
        target = os.path.join(APP_ROOT, 'Datasets/')
        for upload in request.files.getlist("file"):
            filename = upload.filename
            destination = "/".join([target, filename])
            upload.save(destination)
        return render_template('amsg.html', msg=' Dataset Uploaded successfully ', color='bg-success')
@app.route('/ViewDataset')
def ViewDataset():
    data = pd.read_csv('Datasets/kidney_disease (1).csv')
    print(type(data))
    List=data.values.tolist()
    print(type(List))
    return render_template("ViewDataset.html",List=List)

@app.route('/ViewDataset1')
def ViewDataset1():
    data = pd.read_csv('Datasets/kidney_disease (1).csv')
    print(type(data))
    List=data.values.tolist()
    print(type(List))
    return render_template("ViewDataset1.html",List=List)

@app.route('/ViewUsers')
def ViewUsers():
    try:
        result = cursor.execute("select * from userreg")
        UserDetails = cursor.fetchall()
        print(UserDetails)
        conn.commit()
        if result > 0:
            return render_template('userDetails.html',UserDetails=UserDetails)
        else:
            return render_template('amsg.html', msg='User Details Not Available', color='bg-danger')
    except Exception as e:
        return render_template('mmsg.html', msg=str(e), color='bg-danger')


@app.route('/PredictDisease')
def PredictDisease():
    return render_template("predictDisease.html")

@app.route('/PredictDisease1',methods=['post'])
def pYield1():

    SpecificGravity  = request.form.get('SpecificGravity')
    RedBlood = request.form.get('RedBlood')
    CellClumps = request.form.get('CellClumps')
    Bacteria =  request.form.get('Bacteria')
    BloodGlucose = request.form.get('BloodGlucose')
    Haemoglobin = request.form.get('Haemoglobin')
    PackedCell = request.form.get('PackedCell')
    WhiteBlood = request.form.get('WhiteBlood')
    RedBloodCount = request.form.get('RedBloodCount')
    Hypertension = request.form.get('Hypertension')
    Mellitus = request.form.get('Mellitus')
    CoronaryArtery = request.form.get('CoronaryArtery')
    Appetite = request.form.get('Appetite')
    PedalEdema = request.form.get('PedalEdema')
    lmodels = request.form.get('lmodels')

    if lmodels == 'DecisionTreeModel':
        print("DecisionTree")
        with open('./SavedModels/Dt.pickle', 'rb') as f:
            model = pickle.load(f)

        # Prediction

        # Prediction
        k = np.array([[SpecificGravity,RedBlood,CellClumps,Bacteria,BloodGlucose,Haemoglobin,PackedCell,WhiteBlood,RedBloodCount,Hypertension,Mellitus,CoronaryArtery,Appetite,PedalEdema]])
        # k=np.array([[2,1,1,1,1,15,50,15000,5,2,2,2,2,2]])

        predict_dt = model.predict(k)
        predict_dt = int(predict_dt.item())
        # sclr=np.squeeze(predict_dt)
        classes = np.array(['Normal', 'Kidney disease detected'])
        predict_dt
        result = classes[predict_dt]
        print(classes[predict_dt])

    elif lmodels == 'RandomForestModel':
        print("Random Model")
        with open('./SavedModels/RF.pickle', 'rb') as f:
            model = pickle.load(f)

        # Prediction

        # Prediction
        k = np.array([[SpecificGravity,RedBlood, CellClumps, Bacteria, BloodGlucose, Haemoglobin, PackedCell, WhiteBlood, RedBloodCount, Hypertension, Mellitus, CoronaryArtery, Appetite,PedalEdema]])

        predict_dt = model.predict(k)
        predict_dt = int(predict_dt.item())
        # sclr=np.squeeze(predict_dt)
        classes = np.array(['Normal', 'Kidney disease detected'])
        predict_dt
        print(classes[predict_dt])
        result = classes[predict_dt]

    elif lmodels == 'SvmModel':
        print("Svm Model")
        with open('./SavedModels/svc.pickle', 'rb') as f:
            model = pickle.load(f)

        # Prediction

        # Prediction
        k = np.array([[SpecificGravity, RedBlood, CellClumps, Bacteria, BloodGlucose, Haemoglobin, PackedCell,
                       WhiteBlood, RedBloodCount, Hypertension, Mellitus, CoronaryArtery, Appetite,PedalEdema]])

        predict_dt = model.predict(k)
        predict_dt = int(predict_dt.item())
        # sclr=np.squeeze(predict_dt)
        classes = np.array(['Normal', 'Kidney disease detected'])
        predict_dt
        print(classes[predict_dt])
        result = classes[predict_dt]
    elif lmodels == 'knnModel':
        print("knn Model")
        with open('./SavedModels/knn.pickle', 'rb') as f:
            model = pickle.load(f)
        # k = np.array([[1.017408, 1, 0, 0, 99, 11.7, 48, 5000, 2.5, 0, 1, 1, 1, 0]])
        k = np.array([[SpecificGravity,RedBlood, CellClumps, Bacteria, BloodGlucose, Haemoglobin, PackedCell,
                       WhiteBlood, RedBloodCount, Hypertension, Mellitus, CoronaryArtery, Appetite,PedalEdema]])

        predict_dt = model.predict(k)
        predict_dt = int(predict_dt.item())
        # sclr=np.squeeze(predict_dt)
        classes = np.array(['Normal', 'Kidney disease detected'])
        predict_dt
        print(classes[predict_dt])
        result = classes[predict_dt]

    return render_template('Result.html', lmodel=lmodels, result=result)

if __name__ == '__main__':
    app.run(debug=True)
