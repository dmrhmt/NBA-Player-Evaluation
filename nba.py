class Regressions:
    
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    class PolyReg:
        def predict(self):
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            y_train_sc = sc.fit_transform(y_train)
               
            X = x_train.values
            
            poly_reg = PolynomialFeatures(degree = 4)
            x_poly = poly_reg.fit_transform(X)
            #sutunlar sirasiyla x^0, x^1, x^2 olacak sekilde yazdiriyor
            lin_reg2 = LinearRegression()
            #x_poly ile y'yi ogren
            lin_reg2.fit(x_poly, y_train_sc)
            pred = poly_reg.fit_transform(x_test)
            return lin_reg2.predict(pred, sc.inverse_transform(pred))
    
            

    class SVReg:
        def __init__(self, kernel):
            self.kernel = kernel        
        def predict(self):
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVR
            X = x_train.values
            Y = y_train.values
            X_test = x_test.values
            sc = StandardScaler()
            x_train_olcekli = sc.fit_transform(X)
            sc2 = StandardScaler()
            y_train_olcekli = sc2.fit_transform(Y)
            sc3 = StandardScaler()
            x_test_olcekli = sc3.fit_transform(X_test)
            #rbf: gaussian RADIAL BASIS FUNC
            svr_reg = SVR(kernel=self.kernel)
            svr_reg.fit(x_train_olcekli, y_train_olcekli)
            pred = svr_reg.predict(x_test_olcekli)
            return (pred, sc2.inverse_transform(pred))
    
    class DecisionTreeReg:
        def predict(self):
            X = x_train.values
            Y = y_train.values
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            Y = sc.fit_transform(Y)
            
            
            from sklearn.tree import DecisionTreeRegressor
            r_dt = DecisionTreeRegressor(random_state=0)
            r_dt.fit(X,Y)
            pred = r_dt.predict(x_test)
            return (pred, sc.inverse_transform(pred))

    class RandomForestReg:
        def predict(self):        
            X = x_train.values
            Y = y_train.values
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            sc.fit_transform(Y)
            
            #n_estimators: number of decision tree
            rf_reg = RandomForestRegressor(n_estimators=100  , random_state=0)
            rf_reg.fit(X,Y)
            pred = rf_reg.predict(x_test)
            return pred, sc.inverse_transform(pred)



class KMeans:
    def __init__(self, cluster_num, initializer, rand_state, x_train, x_test):
        self.cluster_num = cluster_num
        self.initializer = initializer
        self.rand_state = rand_state
        self.x_train = x_train
        self.x_test = x_test
    def pred(self):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters = self.cluster_num, init = self.initializer, random_state = self.rand_state)
        km_fit = km.fit(self.x_train)
        predict = km_fit.predict(self.x_test)
        return predict


def valuate_player(player_salary, regression_salary):
    if player_salary > regression_salary:
        return("PLAYER IS OVERPAID")
    elif player_salary < regression_salary:
        return("PLAYER IS UNDERPAID")
    else:
        return("PLAYER EARNS EXACT AMOUNT")
    

def preprocessing(data):
    d = cleaning(data)
    d = encoding(d)
    return d

def cleaning(data):
    data = data.drop(columns = ["blanl", "blank2", "Player Name", "#", "Season Start"], axis = 1)
    data = data.dropna()
    
    data["Player Salary"] = data["Player Salary"].astype(int)
    print("salary veri tipi:")
    print(data.dtypes)
    return data

def encoding(data):
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()
    data["Pos"] = le.fit_transform(data["Pos"])
    
    le = None
    le = LabelEncoder()
    data["Tm"] = le.fit_transform(data["Tm"])
    data = data.reset_index()
    return data

def standardilization(data):
    from sklearn.preprocessing import StandardScaler 
    sc = StandardScaler()
    return(sc.fit_transform(data))


import numpy as np
import pandas as pd
veriler = pd.read_excel("basic2.xlsx")


import matplotlib.pyplot as plt
import seaborn as sb

# "AST", "STL", "BLK" dahil rand forest r2 = 0.440360765704644
# + ORB dahil rand forest r2 = 
#  + 3P, 3PA = 0.44653441814150485
#   + OBPM + WS/48 + OWS  = 
#     
#
veriler = preprocessing(veriler)  
veriler = veriler.drop(columns = ["USG%", "index"], axis = 1)

#veriler["Player Salary"], y_unscaled = standardilization(veriler.iloc[:,0:1])
#veriler["MP"] = standardilization(veriler.iloc[:,6:7])
X = veriler.iloc[:,1:]
y = veriler.iloc[:,0].to_frame()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
x_train, x_test, y_train, y_test = x_train.reset_index(drop = True), x_test.reset_index(drop = True), y_train.reset_index(drop = True), y_test.reset_index(drop = True)
y_train, y_test = y_train.astype(float), y_test.astype(float)


reg = Regressions(x_train, y_train, x_test)
"""
poly = reg.PolyReg()
pred_poly_reg = poly.predict()
"""


sv_reg = reg.SVReg("rbf")
pred_sv_reg, pred_sv_reg_uns = sv_reg.predict()

dec_reg = reg.DecisionTreeReg()
pred_dec_tree_reg, pred_dec_tree_uns = dec_reg.predict()

rand_for_reg = reg.RandomForestReg()
pred_rf_reg, pred_rf_reg_uns = rand_for_reg.predict()

print("K Means Cluster test") #rbf harici baska kernel de dene!
y_pred_kmeans = KMeans(5, "k-means++", rand_state = 0, x_train = x_train, x_test = x_test).pred()



"""
classifier = Classifiers(x_train, y_train)

log_cl = classifier.LogisticRegClassifier(x_test)
pred_log_cl = log_cl.predict()

knn_cl = classifier.KNNClassifier(x_test)
pred_knn_cl = knn_cl.predict()


rf = classifier.RandomForestClassifier("gini", x_test)
pred_rf_cl = rf.predict()
"""


from sklearn.metrics import r2_score
"""
print("poly reg r2 degeri:")
print(r2_score(y_test, pred_poly_reg))
"""

print("sv reg r2 degeri:")
print(r2_score(y_test, pred_sv_reg))



print("dec_tree reg r2 degeri:")
print(r2_score(y_test, pred_dec_tree_reg))

print("rand forest reg r2 degeri:")
print(r2_score(y_test, pred_rf_reg))


#BURADAN DEVAM!!!!!!!!!!!!!!!!!!!!!!!!!!!!111


# OYUNCU DEGERLENDIRMESI
x_test_pl = veriler.iloc[50:51,1:]
ply_sal = veriler.iloc[50,0]
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
std_sc = StandardScaler()
std_sc.fit_transform(y_train)

#n_estimators: number of decision tree
rf_reg = RandomForestRegressor(n_estimators=100  , random_state=0)
rf_reg.fit(x_train,y_train)
pred_pl = rf_reg.predict(x_test_pl)
print(pred_pl)
print(valuate_player(ply_sal, pred_pl))


"""
print("log reg classifier r2 degeri:")
print(r2_score(y_test, pred_log_cl))

print("knn classifier r2 degeri:")
print(r2_score(y_test, pred_knn_cl))

print("rand forest classifier r2 degeri:")
print(r2_score(y_test, pred_rf_cl))
"""

import statsmodels.api as sm
# 14 tane, 1 boyutlu, icinde 1 (int tipinde) olan array. Beta0 degerleri
X1 = np.append(arr = np.ones((7901,1)).astype(int), values = X, axis = 1)

X1_l = veriler.iloc[:,2:]
# endog = y(bagimli degisken), exog = x'ler(bagimsiz degiskenler)
r_ols = sm.OLS(endog = y.astype(float), exog = X1_l.astype(float))
r = r_ols.fit()
print(r.summary())


C_mat = veriler.corr()
fig = plt.figure(figsize = (30,30))

sb.heatmap(C_mat, annot = True, xticklabels = True, yticklabels = True)
plt.show()

"""
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

NN_model.fit(X, y, epochs=500, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)
"""



import win32api, winsound
winsound.Beep(440,700)
win32api.MessageBox(0, valuate_player(ply_sal, pred_pl), 'ML Program', 0x00001000)
