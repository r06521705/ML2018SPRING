import sys
import numpy as np
from math import log, floor


def load_predict_ans(k_path , l_path):
    keras_ans = np.load(k_path)
    logis_ans = np.load(l_path)
    keras_ans = np.array(keras_ans)
    logis_ans = np.array(logis_ans)
    keras_ans = keras_ans.reshape(len(keras_ans),1)
    logis_ans = logis_ans.reshape(len(logis_ans),1)
    predict = np.concatenate(( keras_ans,logis_ans), axis=1)
    
    return predict 

def load_label( y_path):
    label = np.load(y_path)
    label = np.array(label)
    label = label.reshape(len(label),)
    
    return label
    


def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 0.00000000000001, 0.99999999999999)

def shuffle(X, Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])


def train(X_train_normed, Y_train):
	# parameter initiallize
    w = np.ones((len(X_train_normed[0])))
    b = np.ones((1,))
    lamda = 10

    l_rate = 0.0001
    epoch_num = 1000
    batch_size = 400
    train_data_size = X_train_normed.shape[0]
    batch_num = int(floor(train_data_size / batch_size))
    display_num = 20
	# train with batch
    for epoch in range(epoch_num):
		# random shuffle
        X_train_normed, Y_train = shuffle(X_train_normed, Y_train)
        epoch_loss = 0.0
        for idx in range(batch_num):
             X = X_train_normed[idx*batch_size:(idx+1)*batch_size]
             Y = Y_train[idx*batch_size:(idx+1)*batch_size]
			
             z = np.dot(X, np.transpose(w)) + b
             y = sigmoid(z)
			
             cross_entropy = -(np.dot(Y, np.log(y)) + np.dot((1 - Y), np.log(1 - y)))
             epoch_loss += cross_entropy
			
             w_grad = np.sum(-2 * X * (Y - y).reshape((batch_size,1)), axis=0) 
             b_grad = np.sum(-2 * (Y - y))

             #for i in range(w_grad.size):
              #   w_lr[i] = w_lr[i] + w_grad[i]**2
               #  if w_lr[i] != 0:
                #     w[i] = w[i] - (l_rate / np.sqrt(w_lr[i])) * w_grad[i]    
                  
             #b_lr = b_lr + b_grad**2
             #b = b - l_rate /np.sqrt(b_lr)* b_grad
             w = w - l_rate * w_grad
             b = b - l_rate * b_grad

        if (epoch+1) % display_num == 0:
            print ('avg_loss in epoch%d : %f' % (epoch+1, (epoch_loss / train_data_size)))
            

    return w, b



predict = load_predict_ans("keras_ans.npy" , "logis_ans.npy")
label = load_label("yfortrain.npy")

w , b = train(predict, label)   # w = [5.56490382 , 0.813641]  ==> keras model得出來的比重比較大 , logestic的比較小

test_ensemble = load_predict_ans("keras_test.npy" ,"logis_test.npy" )

output = np.dot(test_ensemble , np.transpose(w)) + b

final = sigmoid(output)
final_ = np.around(final)
with open("ensemble.csv", 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(final_):
        f.write('%d,%d\n' %(i+1, v))





