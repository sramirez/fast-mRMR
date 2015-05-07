from random import randint  
BINS = 10
def create_doc(name, ds, fs, bins):
	f = open(name, 'w')
	for j in range(0, fs -1):
	    f.write('f' + str(j) + ',' )
	f.write('f' + str(ds) + '\n' )
	     
	for i in range(0, ds):
	    for j in range(0, fs -1):
		f.write(str(randint(0,bins))+ ',')
	    f.write(str(randint(0,bins))+ '\n')
	f.close()

"""create_doc('python_features_100', 80, 100, 10)
create_doc('python_features_1000', 80, 1000, 10)
create_doc('python_features_10000', 80, 10000, 10)
create_doc('python_features_100000', 80, 100000, 10)"""
create_doc('python_samples_400K_2K_15.csv', 400000, 2000, 15)
