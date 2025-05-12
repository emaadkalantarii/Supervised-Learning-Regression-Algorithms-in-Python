# ========================== Supervised Learning (SL) - Regression ==========================
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

# 1-data aquuisition
main_dataset=pd.read_excel('./Data Take Home Assignment 1 Exercise A.xlsx')
starts_row = ( 17 - 1 ) * 20 + 2
ends_row = 17 * 20 + 1
dataset_g_17 = main_dataset.iloc[starts_row:ends_row+1,:]

# 2-data transformation (min max normalization)
dataset_g_17['X']=dataset_g_17['X'].apply(lambda x : (x-min(dataset_g_17['X']))/(max(dataset_g_17['X'])-min(dataset_g_17['X'])))
dataset_g_17['Y']=dataset_g_17['Y'].apply(lambda x : (x-min(dataset_g_17['Y']))/(max(dataset_g_17['Y'])-min(dataset_g_17['Y'])))   

# 3-least square
# Y=A+BX
# SSE=[0.21-(A+0.9*B)]^2 + [0.11-(A+0.63*B)]^2 + ... + [0-(A+1*B)]^2
x=np.asanyarray(dataset_g_17[['X']])
y=np.asanyarray(dataset_g_17[['Y']])
# calculate the parameters of A and B by doing partial derivatives:
A = 0.864
B = -0.928
#plot
plt.scatter(x,y)
plt.plot(x , B*x + A , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

# 5-gradient descent - first iter
a,b = 1 , -0.8
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']
dataset_g_17['J']=0.5*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**2
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])
dataset_g_17['∂J/∂b']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])*dataset_g_17['X']
total_j=dataset_g_17['J'].sum()
a_new=a-0.01*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.01*dataset_g_17['∂J/∂b'].sum()
plt.scatter(x,y)
plt.plot(x , b_new*x + a_new , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

#6-second iter
a,b=0.96 , -0.82
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']
dataset_g_17['J']=0.5*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**2
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])
dataset_g_17['∂J/∂b']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])*dataset_g_17['X']
total_j=dataset_g_17['J'].sum()
a_new=a-0.01*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.01*dataset_g_17['∂J/∂b'].sum()
plt.scatter(x,y)
plt.plot(x , b_new*x + a_new , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

# 7-third iter
a,b = 0.93 , -0.84
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']
dataset_g_17['J']=0.5*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**2
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])
dataset_g_17['∂J/∂b']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])*dataset_g_17['X']
total_j=dataset_g_17['J'].sum()
a_new=a-0.01*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.01*dataset_g_17['∂J/∂b'].sum()
plt.scatter(x,y)
plt.plot(x , b_new*x + a_new , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

# 8-Fourth iter
a,b = 0.91 , -0.85
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']
dataset_g_17['J']=0.5*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**2
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])
dataset_g_17['∂J/∂b']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])*dataset_g_17['X']
total_j=dataset_g_17['J'].sum()
a_new=a-0.01*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.01*dataset_g_17['∂J/∂b'].sum()

plt.scatter(x,y)
plt.plot(x , b_new*x + a_new , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

# 8-last iter
a,b=0.89 , -0.86
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']
dataset_g_17['J']=0.5*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**2
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])
dataset_g_17['∂J/∂b']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])*dataset_g_17['X']
total_j=dataset_g_17['J'].sum()
a_new=a-0.01*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.01*dataset_g_17['∂J/∂b'].sum()
plt.scatter(x,y)
plt.plot(x , b_new*x + a_new , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

# polynomial regression
# Preprocessing the data in order to make them polynomial
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_poly=poly.fit_transform(x)
clf=linear_model.LinearRegression()
y_poly=clf.fit(x_poly,y)
#y=a+bx+cx2
a=clf.intercept_[0]
b=clf.coef_[0][1]
c=clf.coef_[0][2]
plt.scatter(x,y)
xx=np.arange(0,1,0.01)
yy=a+b*xx+c*np.power(xx , 2)
plt.plot(xx,yy , color='red')
plt.show()

#first iteration
a,b,c=1 , -2 , 1
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']+c*np.power(dataset_g_17['X'] , 2)
dataset_g_17['J']=0.25*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**4
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3
dataset_g_17['∂J/∂b']=-dataset_g_17['X']*((dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3)
dataset_g_17['∂J/∂c']=-2*c*dataset_g_17['X']*((dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3)
total_j=dataset_g_17['J'].sum()
a_new=a-0.5*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.5*dataset_g_17['∂J/∂b'].sum()
c_new=c-0.5*dataset_g_17['∂J/∂c'].sum()
plt.scatter(x,y)
xx=np.arange(0,1,0.01)
plt.plot(xx , a_new+b_new*xx+c_new*np.power(xx , 2) , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


#second iteration
a,b,c= 1.03, -1.98 , 1.04
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']+c*np.power(dataset_g_17['X'] , 2)
dataset_g_17['J']=0.25*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**4
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3
dataset_g_17['∂J/∂b']=-dataset_g_17['X']*((dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3)
dataset_g_17['∂J/∂c']=-2*c*dataset_g_17['X']*((dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3)
total_j=dataset_g_17['J'].sum()
a_new=a-0.5*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.5*dataset_g_17['∂J/∂b'].sum()
c_new=c-0.5*dataset_g_17['∂J/∂c'].sum()
plt.scatter(x,y)
xx=np.arange(0,1,0.01)
plt.plot(xx , a_new+b_new*xx+c_new*np.power(xx , 2) , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()


#third iteration
a,b,c= 1.05, -1.97 , 1.06
dataset_g_17['Y_predicted']=a+b*dataset_g_17['X']+c*np.power(dataset_g_17['X'] , 2)
dataset_g_17['J']=0.25*(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**4
dataset_g_17['∂J/∂a']=-(dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3
dataset_g_17['∂J/∂b']=-dataset_g_17['X']*((dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3)
dataset_g_17['∂J/∂c']=-2*c*dataset_g_17['X']*((dataset_g_17['Y']-dataset_g_17['Y_predicted'])**3)
total_j=dataset_g_17['J'].sum()
a_new=a-0.5*dataset_g_17['∂J/∂a'].sum()
b_new=b-0.5*dataset_g_17['∂J/∂b'].sum()
c_new=c-0.5*dataset_g_17['∂J/∂c'].sum()
print(a_new,b_new,c_new,total_j)
plt.scatter(x,y)
xx=np.arange(0,1,0.01)
plt.plot(xx , a_new+b_new*xx+c_new*np.power(xx , 2) , color='red')
plt.ylabel('Y')
plt.xlabel('X')
plt.show()