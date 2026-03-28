# def identify(n):
#     if n==0:
#         print("neutral")
#     elif n<0:
#         print("negative")
#     else:
#         print("Positive")
# identify(-4.6)

# x=input("enter x value")
# y=input("enter y value")
# z=input("enter the operation")
# def calculator(z):
#     if z=='+':
#         print(x+y)
#     elif z=='*':
#         print(x*y)
#     elif z=='/':
#         print(x/y)
#     else:
#         print("you stupid!, don you no maths these are only available operators")
# calculator(z)

# a=['apple','orange','guava']
# print(a[1])

a=[{'employee':'venkat','department':'AIML','ID':1234},
{'employee':'raj','department':'cse','ID':2345},
{'employee':'prav','department':'ece','ID':3456},             
{'employee':'vam','department':'civ','ID':4567}]
# print(type(a))
# print(a[2])


import pandas as pd
# a=[{'name': ['a','b','c','d']},
#    {'department':['AIML','cse','ece','civ']},
#    {'ID':[1234,2345,3456,4567]}]

df=pd.DataFrame(a)
dp=df[['employee','ID']]
# sr=df.iloc[0]
# sr1=df.iloc[1:3]
# lb=df.loc[0]
# print(sr)
# print(sr1)
# print(lb)
fl=df[df['ID']>1000]
op=df[df['department']== 'cse']
ui=df[(df['ID']>1000) | (df['department']== 'cse')]
df['salary']=[10000,20000,30000,40000]
df['salary_bonus']=df['salary']*0.2
df['employee_upper']=df['employee'].str.upper()
print(df)
  
i am meghana 
meghna git hub