import math
import csv 
import numpy as np
import sys


#------------處理資料-----------------------------
data = [] #儲存汙染物資料
for i in range(18):
    data.append([]) #創出18個儲存汙染物的空間

countinrow = 0 
temp = open("train.csv" , 'r' ,encoding='big5') 
databox = csv.reader(temp , delimiter=",")
for i in databox:
    if countinrow != 0 : #第一行開頭沒有資料
        for j in range(3,27):
           if i[j] == "NR": #如果碰到nr先設為0
               data[(countinrow-1) % 18].append(float(0))
           else:
               data[(countinrow-1) % 18].append(float(i[j]))
    countinrow = countinrow +1
temp.close()




#------------創建x,y資料--抽取features-----------

x = [] #丟入的數據
y = [] #實際pm2.5的值
   
for i in range(12): #從每個月來看，每9小時為input(X),第10小時的pm2.5值為y,這樣取每個月共會有471筆資料     
    for j in range(471):
        x.append([])
        for k in range(18): #18種汙染物
            for m in range(9): #連續9小
                x[471*i+j].append(data[k][480*i+j+m])
        y.append([])
        y[471*i+j].append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)
      
#--------------------add bias---------------
xbias = np.ones((len(x) , 1))
x = np.concatenate((xbias , x), axis = 1) 

#---------------------建立model  training--------------
w = np.zeros((len(x[0]) , 1))

learningrate = 0.000000000207
#-------------------------
repeat = 100000

x_trans = x.transpose()
sec_der = np.zeros((len(x[0]),1))
for i in range(repeat):
    predict_y = np.dot(x , w)
    loss = predict_y - y 
    totalloss = np.sum(loss**2) / len(y)
    totalloss_sqrt = math.sqrt(totalloss)
    gra = np.dot(x_trans , loss) * 2
    sec_der += gra**2
    adag = np.sqrt(sec_der)
    w = w - learningrate * gra
    print("times : %d , totalloss : %f" % (i ,totalloss_sqrt ))
    
#------儲存找到的參數係數w--------------------------
np.save('model_hw1.npy',w)
w = np.load('model_hw1.npy')

#------先處裡testing data---------------
x_test=[]
data = [] #儲存汙染物資料
for i in range(18):
    data.append([]) #創出18個儲存汙染物的空間

countinrow = 0 
temp = open("test.csv", 'r' ,encoding='big5') 
databox = csv.reader(temp , delimiter=",")
for i in databox:
    for j in range(2,11):
        if i[j] == "NR": #如果碰到nr先設為0
            data[(countinrow) % 18].append(float(0))
        else:
            data[(countinrow) % 18].append(float(i[j]))
    countinrow = countinrow +1
temp.close()
for i in range(260):
    x_test.append([])
    for j in range(18):
        for k in range(9):
            x_test[i].append(data[j][i*9+k]) #x_test
x_test = np.array(x_test)            
      
#--------------------add bias---------------
xbias = np.ones((len(x_test) , 1))
x_test = np.concatenate((xbias , x_test), axis = 1) 

#---------------------------------丟入測試資料------------------
answer = []
for i in range(len(x_test)):
    answer.append(["id_"+str(i)])
    ans = np.dot(x_test[i] , w)
    answer[i].append(ans[0])
    
#----------------------------------寫出csv檔--------------------
filename = "predict.csv"
temp = open(filename, 'w' , newline='')
exc = csv.writer(temp)
exc.writerow(["id","value"])
for i in range(len(answer)):
    exc.writerow(["id_"+str(i),answer[i][1]])  
