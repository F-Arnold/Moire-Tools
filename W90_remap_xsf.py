'''
Author: Florian Arnold
Date: September 26, 2025
Purpose: 
    Wannier90 files export xsf files of Wannier functions in large supercells for the 3D datagrid.
    This makes plotting harder and results in huge file sizes.
    With this script, all data is mapped into the same unit cell as is used for the atoms.
    For example, a file with a 5x7x1 plotting supercell is reduced from 129 MB to about 3.3 MB, a factor of almost 40.
Usage: run in directory where xsf files are stored
Disclaimer: 
    Only tested for a limited number of systems for now! 
    Always compare some original and mapped files for each new system and new supercell setup, please.
'''


### import modules
import sys # system operations
import os
import numpy as np

### constants
PRECISION = 8 # number of decimals to write for floats of datagrid
DISABLE_X_MAPPING = False # toggle mapping in x-direction
DISABLE_Y_MAPPING = True # toggle mapping in y-direction
DISABLE_Z_MAPPING = False # toggle mapping in z-direction

### iterate over all xsf files in directory

for filename in os.listdir(os.getcwd()):
    if filename.endswith(".xsf") and "mapped" not in filename: # avoid already mapped files
        filename = filename.replace('.xsf', '')
        print(filename)
        
        ### extract main data from file

        ## open file
        with open(filename+'.xsf', 'r') as f:
            lines = f.readlines()
        ## read unit cell data
        # find the PRIMVEC block
        for i, line in enumerate(lines):
            if line.strip() == "PRIMVEC":
                vec1 = np.array(list(map(float, lines[i + 1].strip().split())))
                vec2 = np.array(list(map(float, lines[i + 2].strip().split())))
                vec3 = np.array(list(map(float, lines[i + 3].strip().split())))
                break
        else:
            raise ValueError("PRIMVEC block not found.")
        # combine into 3x3 numpy array
        cell = np.array([vec1, vec2, vec3])

        ## read grid for 3D data
        # find header and parse it
        for i, line in enumerate(lines):
            if line.strip() == "BEGIN_DATAGRID_3D_UNKNOWN":
                dim_line = lines[i + 1]
                index_dim_line = i+1 # needed for export later
                origin_line = lines[i + 2]
                vec1_line = lines[i + 3]
                vec2_line = lines[i + 4]
                vec3_line = lines[i + 5]
                data_start_index = i + 6
                break
        else:
            raise ValueError("BEGIN_DATAGRID_3D_UNKNOWN block not found.")
        # parse dimensions
        nx, ny, nz = map(int, dim_line.strip().split())
        # parse origin and lattice vectors
        origin = np.array(list(map(float, origin_line.strip().split())))
        vec1 = np.array(list(map(float, vec1_line.strip().split())))
        vec2 = np.array(list(map(float, vec2_line.strip().split())))
        vec3 = np.array(list(map(float, vec3_line.strip().split())))

        ## read data entries, map to coordinates
        # read all scalar values
        raw_values = ' '.join(lines[data_start_index:]).split()
        raw_values = raw_values[0:-2] # last two values are keywords => remove
        if len(raw_values) != nx * ny * nz:
            raise ValueError("Number of scalar values does not match grid dimensions.")
        values = np.array(list(map(float, raw_values)))
        values = values.reshape((nz, ny, nx)).transpose((2, 1, 0))  # shape: (nx, ny, nz), weird code as order in file is z->y->x
                    
        ### map values to primitive unit cell

        ## define new resolution
        # get number of unit cells in each direction
        x = ((nx+1)/nx) * np.linalg.norm(vec1) / np.linalg.norm(cell[0])
        y = ((ny+1)/ny) * np.linalg.norm(vec2) / np.linalg.norm(cell[1])
        z = ((nz+1)/nz) * np.linalg.norm(vec3) / np.linalg.norm(cell[2])
        scale_x, scale_y, scale_z = round(x), round(y), round(z)
        print("   scaling", x, y, z)
        print("   factor", scale_x, scale_y, scale_z)
        # diable mapping in requested directions
        if DISABLE_X_MAPPING: scale_x = 1
        if DISABLE_Y_MAPPING: scale_y = 1
        if DISABLE_Z_MAPPING: scale_z = 1
        # reduce gridpoints as needed
        print("   old grid", nx, ny, nz)
        nx = int(nx/scale_x)
        ny = int(ny/scale_y)
        nz = int(nz/scale_z)

        ## map to primitive unit cell
        # create empty array for storing values
        mapped = np.zeros((nx, ny, nz), dtype=float)
        # map original grid to reduced grid (vectorized version)
        reshaped = values.reshape(
            scale_x, nx,
            scale_y, ny,
            scale_z, nz
        ).transpose(1, 3, 5, 0, 2, 4)
        mapped = reshaped.sum(axis=(3, 4, 5))
        # map original grid to reduced grid (old, factor 1000 slower version)
        # for i in range(nx):
            # for j in range(ny):
                # for k in range(nz):
                    # for x in range(scale_x):
                        # for y in range(scale_y):
                            # for z in range(scale_z):
                                # mapped[i, j, k] += values[i+x*nx, j+y*ny, k+z*nz]
        # shift all elements to one index less along each axis to map origin to zeros
        if not DISABLE_X_MAPPING: mapped = np.roll(mapped, shift=-1, axis=0)  # shift along x
        if not DISABLE_Y_MAPPING: mapped = np.roll(mapped, shift=-1, axis=1)  # shift along y
        if not DISABLE_Z_MAPPING: mapped = np.roll(mapped, shift=-1, axis=2)  # shift along z
        # handle PBC correctly
        first_x = mapped[0:1, :, :] # shape (1, ny, nz)
        mapped = np.concatenate((mapped, first_x), axis=0)
        first_y = mapped[:, 0:1, :] # shape (nx+1, 1, nz)
        mapped = np.concatenate((mapped, first_y), axis=1)
        first_z = mapped[:, :, 0:1] # shape (nx+1, ny+1, 1)
        mapped = np.concatenate((mapped, first_z), axis=2)
        # change of PBC definition needs increase of gridpoint variable by one each
        if not DISABLE_X_MAPPING: nx += 1
        if not DISABLE_Y_MAPPING: ny += 1
        if not DISABLE_Z_MAPPING: nz += 1
        print("   new grid", nx, ny, nz)
        # adjust origin: move to zero to map grid onto unit cell (for those that should be mapped)
        if not DISABLE_X_MAPPING: origin[0] = 0 # x-direction
        if not DISABLE_Y_MAPPING: origin[1] = 0 # y-direction
        if not DISABLE_Z_MAPPING: origin[2] = 0 # z-direction
        # take new grid vectors for datagrid from unit cell (for those that should be mapped)
        if not DISABLE_X_MAPPING: vec1 = np.asarray(cell[0])
        if not DISABLE_Y_MAPPING: vec2 = np.asarray(cell[1])
        if not DISABLE_Z_MAPPING: vec3 = np.asarray(cell[2])
        
        ### export
        
        ## write new xsf file
        # create new file
        file = open(filename+'_mapped.xsf', 'w+')
        # write comment that mapped information
        file.write("       # Data mapped onto crystal unit cell\n")
        # write information that remains identical
        for i in range(index_dim_line): # use index determined above
            file.write(lines[i])
        # write grid
        file.write(str(nx)+'   '+str(ny)+'   '+str(nz)+'\n')
        # write origin and vectors
        file.write(str(origin[0])+'   '+str(origin[1])+'   '+str(origin[2])+'\n')
        file.write(str(vec1[0])+'   '+str(vec1[1])+'   '+str(vec1[2])+'\n')
        file.write(str(vec2[0])+'   '+str(vec2[1])+'   '+str(vec2[2])+'\n')
        file.write(str(vec3[0])+'   '+str(vec3[1])+'   '+str(vec3[2])+'\n')
        # prepare values for writing
        export = []
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    export.append(mapped[i,j,k])
        # write actual values
        line = ''
        for i in range(0, len(export)):
            line += ' ' + format(export[i], '.'+str(PRECISION)+'f')
            if (i+1)%6==0: # 6 elements per line
                file.write(line+'\n')
                line = '' # initialize new line
        # finish export 
        file.close()

