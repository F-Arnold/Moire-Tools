'''
Analyze bond lengths in 1D structures
Author: Florian Arnold
Date: September 28, 2023
'''

### import
import os # change directory
import numpy as np # dealing with arrays more efficient
import matplotlib.pyplot as plt # create plots

### constants
DIST = 1.8 # Angstrom, maximum distance to handle an atom as neighbor
HIST_STEP = 0.001 # [Ang], step size in histogram for bond length

### iterate over all .xyz files
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".xyz"):
        filename = filename.replace('.xyz', '')
        print(filename)
        
        ### read content of file
        # import file
        input_file = []
        with open(filename+'.xyz', 'r') as f:
           for line in f:
               input_file.append(line.split())
        # read atoms
        Z = int(input_file[0][0])
        types = [] # atom types (1D list, [type])
        coord = [] # atom coordinates (2D list, [x, y, z])
        for i in range(2, 2+Z):
            types.append(input_file[i][0])
            coord.append([float(input_file[i][1]), float(input_file[i][2]), float(input_file[i][3])])
        # read unit cell
        vec_a = [float(input_file[Z+2][1]), float(input_file[Z+2][2]), float(input_file[Z+2][3])]
        # transform to numpy arrays
        coord = np.asarray(coord)
        vec_a = np.asarray(vec_a)
        
        ### general neighbor search, for all atoms
        ## create a 3x1x1 supercell from coordinates
        types_sc, coord_sc = [], []
        for m in [-1, 0, 1]:
            for i in range(len(coord)):
                coord_sc.append(coord[i] + m*vec_a)
                types_sc.append(types[i])
        coord_sc = np.asarray(coord_sc)
        ## identify all valid bonds in the system (bad scaling, but we have small systems)
        bonds = []
        for i in range(len(coord)):
             for k in range(len(coord_sc)):
                 d = np.sqrt((coord[i][0]-coord_sc[k][0])**2 + (coord[i][1]-coord_sc[k][1])**2 + (coord[i][2]-coord_sc[k][2])**2)
                 if d<DIST and d>0.1: # bond identified
                     bonds.append(sorted([i, k%Z])) # supercell index mapped to primitive cell
        # find unique bonds
        bonds = [list(t) for t in set(tuple(element) for element in bonds)] # https://stackoverflow.com/questions/3819348/removing-duplicate-entries-from-multi-d-array-in-python
        ## for all bonds save bond type and bond length
        bond_types, bond_dist = [], []
        for bond in bonds:
            # bond types
            bond_types.append(sorted([types[bond[0]], types[bond[1]]])) # get types, sorted alphabetically
            # bond lengths
            coord_1, coord_2 = coord[bond[0]], coord[bond[1]] # get coordinates
            dist = []
            for m in [-1, 0, 1]: # include PBC
                dist.append(np.sqrt((coord_1[0]-coord_2[0]+m*vec_a[0])**2 + (coord_1[1]-coord_2[1]+m*vec_a[1])**2 + (coord_1[2]-coord_2[2]+m*vec_a[2])**2))
            bond_dist.append(min(dist))
        ## extract unique bond types
        unique_bonds = [list(t) for t in set(tuple(element) for element in bond_types)] # https://stackoverflow.com/questions/3819348/removing-duplicate-entries-from-multi-d-array-in-python
        
        ### for each bond type: extract results and plot as histogram, save raw data to file
        # iterate over all bond types
        for bond in unique_bonds: # do for each
            ## extract all distance values for current bond type
            dist = []
            for i in range(len(bond_types)):
                if bond==bond_types[i]:
                    dist.append(bond_dist[i])
            dist = sorted(dist) # sort to easier see minimum and maximum in file output
            ## save numbers to file
            file = open(filename + '_bond_' + bond[0] + '-' + bond[1] + '.dat', 'w+')
            for i in range(len(dist)):
                file.write(str(dist[i]) + '\n')
            file.close()
            ## only plot histogram if there are different values
            if max(dist) - min(dist) > HIST_STEP: 
                ## plot histogram of bond length distribution
                # set up plot
                fig, ax = plt.subplots()
                ax.set_xlabel('Bond length [Å]')
                ticks = np.arange(round(min(dist),2), round(max(dist),2)+1.5*HIST_STEP, 10*HIST_STEP) # calculate ticks
                ax.set_xticks(ticks)
                # create histogram
                sequence = np.arange(round(min(dist),2)-10*HIST_STEP, round(max(dist),2)+10*HIST_STEP, HIST_STEP) # create sequence to set up histogram manually
                hist = ax.hist(x=dist, bins=sequence, edgecolor='black', linewidth=1)#, label='distribution')
                # fix issue with y-axis labels
                ticks = ax.get_yticks().tolist()
                for i in range(len(ticks)):
                    ticks[i] = int(round(ticks[i])) # get rid of decimal place
                ticks.append(max(ticks)+1) # make sure to not loose maximum value due to rounding
                ax.set_yticks(ticks)
                # fix issue with x-axis range
                ax.set_xlim(min(dist)-5*HIST_STEP, max(dist)+5*HIST_STEP)
                # add line for average value to plot
                mean = sum(dist)/len(dist)
                ax.axvline(x=mean, color='red', label=str(round(mean,3))+' Å')
                ax.legend()
                # save plot
                fig.savefig(filename + '_bond_' + bond[0] + '-' + bond[1] + '.png')
                plt.close()
        
        
            
        
