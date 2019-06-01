import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

file_path="/media/chandan/chandan/python/csv/digit_recognizer/train.csv"
file_read=pd.read_csv(file_path)

#print(file_read)
X=file_read.loc[:,file_read.columns !='label']
#print(features.head())
#print(X.head())
y=file_read.label
#print(y.head())

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#print(x_train)
#print(y_train)


file_read_test=pd.read_csv("/media/chandan/chandan/python/csv/digit_recognizer/test.csv")
#print(file_read_test)

model=DecisionTreeClassifier(random_state=1)


model.fit(x_train,y_train)
res=model.predict(file_read_test)
#res1=[res]
#print(res[0])
#for  i in res:
 #print(i)
#print(model.score(x_test,y_test))

#total=len(file_read_test)
#print(total)

with open('new.csv','w') as f:
    writer=csv.writer(f)
    writer.writerow(['ImageId','Label'])
    #writer.writerow([1,11])
    for i in range(1,28001):
       p=res[i-1]
       writer.writerow([i,p])


f.closed




