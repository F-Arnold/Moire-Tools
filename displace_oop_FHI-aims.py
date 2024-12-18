### import modules
import random
import os

### constants
DISPLACEMENT = 0.1 # Ang, interval +- to change z coordinates

### iterate over all possible files
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".in"):
        
        ### import input file
        file = open(filename)
        lines = file.readlines()
        file.close()
        
        ### edit coordinate lines
        for i in range(len(lines)):
            if 'atom' in lines[i]: # check for coordinate lines
                line = lines[i].split()
                line[3] = str(float(line[3]) + random.uniform(-DISPLACEMENT, DISPLACEMENT))
                ## overwrite original text
                lines[i] = line[0]
                for k in range(1, len(line)):
                    lines[i] += ' ' + line[k]
                lines[i] += '\n'
                
        ### overwrite original file
        file = open(filename+'_oop', 'w+')
        for i in range(len(lines)):
            file.write(lines[i])
        file.close()
                
        
        