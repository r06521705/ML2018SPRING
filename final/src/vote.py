import csv
import pandas as pd
import numpy as np

me = pd.read_csv('pred/vote_csv/merge2.csv', header=0)
claw = pd.read_csv('pred/vote_csv/merge3.csv', header=0)
third = pd.read_csv('pred/vote_csv/merge1.csv', header=0)

cnt = 0
same_list = []
for i in range(5060):
	if me['ans'][i] == claw['ans'][i] and claw['ans'][i] == third['ans'][i]:
		cnt += 1
	else:
		same_list.append(i)

print(cnt)


f = open('pred/merged_csv/merge.csv','w')
fo = csv.writer(f)
cnt = 0
header = ['id','ans']
fo.writerow(header)

'''
# 偏袒 m2 跟 m3 
for i in range(5060):
	if me['ans'][i] == claw['ans'][i]:
		e = me['ans'][i]
	else:
		if me['ans'][i] == third['ans'][i]:
			e = me['ans'][i]
		if claw['ans'][i] == third['ans'][i]:
			e = claw['ans'][i]
		if me['ans'][i] != third['ans'][i] and claw['ans'][i] != third['ans'][i]:
			#e = claw['ans'][i]
			#e = me['ans'][i]
			e = third['ans'][i]

	s = [cnt,str(e)]
	fo.writerow(s)
	cnt = cnt + 1
print('done')
'''

c1 = 0
c2 = 0
c3 = 0
c4 = 0
c5 = 0

# 三方權重一樣
for i in range(5060):
	if me['ans'][i] == claw['ans'][i] and claw['ans'][i] == third['ans'][i]:
		e = me['ans'][i]
		c1 += 1
	else:

		if me['ans'][i] == claw['ans'][i] and me['ans'][i] != third['ans'][i]:
			e = claw['ans'][i]
			c2 += 1 
		if me['ans'][i] == third['ans'][i] and me['ans'][i] != claw['ans'][i]:
			e = me['ans'][i]
			c3 += 1
		if claw['ans'][i] == third['ans'][i] and me['ans'][i] != claw['ans'][i]:
			e = claw['ans'][i]
			c4 += 1
		if me['ans'][i] != third['ans'][i] and claw['ans'][i] != third['ans'][i] and me['ans'][i] != claw['ans'][i]:
			#e = claw['ans'][i]
			#e = me['ans'][i]
			e = me['ans'][i]
			c5 += 1
	s = [cnt,str(e)]
	fo.writerow(s)
	cnt = cnt + 1
print('done')
print(c1)
print(c2)
print(c3)
print(c4)
print(c5)



