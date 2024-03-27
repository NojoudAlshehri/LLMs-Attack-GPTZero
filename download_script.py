import vrplib

names = ['E-n22-k4']

for name in names: 
    # Download an instance and a solution file 
    vrplib.download_instance(name, "./instances/E/") 
    vrplib.download_solution(name, "./instances/E")