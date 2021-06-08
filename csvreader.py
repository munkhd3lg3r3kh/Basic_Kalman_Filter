import csv
import matplotlib.pyplot as plt

ref = []
sensor = []
with open("example.csv", newline='') as File :
    reader = csv.reader(File)
    for row in reader : 
        if row[0] != "":            
            ref.append(float(row[1]))
            sensor.append(float(row[2]))
    


