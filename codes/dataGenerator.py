'''
Generate the data with concept drift

vector = (x, y, z)

if x + y > threshold => class = 1
if x + y <= threshold => class = 0

8, 9, 7 and 9.5 ; 5,000, 30,000 and 45,000, yielding 60,000
'''

import math, random


threshold = 5
outputFile = 'myData.data'


def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

# randrange_float(2.1, 4.2, 0.3) # returns 2.4

def initializeData():
    with open('myData.data', 'w') as f:
        for i in range(100):
            x = randrange_float(0.0, 10.0, 0.1)
            y = randrange_float(0.0, 10.0, 0.1)

            if x + y > threshold:
                c = 1
            else:
                c = 0
            f.write("%f,%f,%d\n" % (x, y, c))

def createConceptDrift():



