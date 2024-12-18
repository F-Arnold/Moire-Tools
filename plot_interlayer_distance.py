'''
The scritp is meant to plot the interlayer distance landscape of twisted bilayer MoS2 systems.
Author: Florian M. Arnold
Affiliation: TU Dresden
Usage:
    * The script is written in a way that it iterates over all structures files in its directory.
    * It expects the structures to be given in the xsf (XCrySDen) format. 
    * Filenames are assumed to have twist angle plus "deg" -> if you have it differently please adapt the script.
    * The structure is assumed to be 2D, periodic in both x- and y-direction, and with a sufficient vacuum gap in z-direction.
    * If you have another structure format I recommend ASE (Atomic Simulation Environment) for converting.
    * The script uses the "pickle" module to store data to have a faster rerun if required.
    * No support for other atomic systems is currently added, but can be included by modifying the import and sorting routines.
'''

### import modules
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable # matplotlib utility function
from scipy import interpolate # interpolation of data - splines
import pickle # save results variables and lists for later use
import math
import sys


### constants
## script behavior
USE_PICKLES = True # toggle if pickle files (.pkl) should be read (if available) and written (else)
STRONG_CORRUGATION = False # additional routine for sorting when layers are too strongly corrugated -> slow, only activate when needed!
FIXED_COLORBAR = True # toggle if colorbar should have fixed range (set D_MIN and D_MAX accordingly) or a variable range (based on structure)
## interpolation settings
DIST_EXTEND = 35 # Ang, value to extend around simulation cell for interpolation
GRIDDATA_ORDER = 'cubic' # order of griddata interpolation. options: 'nearest', 'linear', 'cubic'
## plotting
SIZE_PT = 0.4 # size of points for scatter plot, this value gave generally good results
D_MIN, D_MAX = 6.03, 6.50 # Ang, interlayer distance references (min & max) to have consistent colorbar
DPI = 300 # dpi for figure export
PLOT_MOIRE_CELL = True # toggle if moiré cell is shown for main interlayer distance plot or not
## numerics
SIN60 = np.sin(60/360*2*np.pi) # often needed for b axis of unit cell
COS60 = np.cos(60/360*2*np.pi) # for nicer looking code
D_Mo_THRESH = 4 # Ang, threshold for Mo-Mo distance in neighbor search
A_ML = 3.186 # Ang, cell constant of relaxed monolayer (used to approximate moiré cell size)



#===============#
# Main function #
#===============#

def main():
    
    ### iterate over all xsf in directory
    directory = os.fsencode('.') # use current working directory
    for file in os.listdir(directory): # iterate over all files
        filename = os.fsdecode(file)
        if filename.endswith(".xsf"):
            filename = filename.replace('.xsf', '') # remove file type
            twist_angle = float(filename.replace('deg', '')) # get twist angle from filename
            print(filename, '('+str(twist_angle)+'°)')
            
            ### import structure
            
            ## check if import routine is needed or results can be taken from pickle
            if USE_PICKLES: # usage requested -> check if available
                try: # open pickle
                    with open(filename+'.pickle', 'rb') as f: var = pickle.load(f)
                    Mo_UL, Mo_LL, cell = var[0], var[1], var[2] # extract constant
                    import_already_done = True
                except: # no existing pickle file or error upon extraction
                    import_already_done = False
            else: # pickle files are explicitly not requested
                import_already_done = False
                
            ## read atoms and cell from file if needed
            if not import_already_done: # results have to be (re)calculated
                cell, atoms = ReadXsf(filename) # import from xsf 
                Mo_UL, Mo_LL = SortAtoms(cell, atoms) # sort into upper layer (UL) and lower layer (LL)
                with open(filename+'.pickle', 'wb') as f:
                    pickle.dump([Mo_UL, Mo_LL, cell], f) # store in pickle file
            
            
            ### calculate interlayer distance for structure
            
            ## check if interpolation routine is needed or results can be taken from pickle
            if USE_PICKLES: # usage requested -> check if available
                try: # open pickle
                    with open(filename+'.pickle', 'rb') as f: var = pickle.load(f)
                    Mo_UL, Mo_LL, cell, x_grid, y_grid, d_local, A_REF = var[0], var[1], var[2], var[3], var[4], var[5], var[6] # extract constant
                    interpolation_already_done = True
                except: # no existing pickle file or error upon extraction
                    interpolation_already_done = False
            else: # pickle files are explicitly not requested
                interpolation_already_done = False
                
            ## perform interpolation if needed
            if not interpolation_already_done: # results have to be (re)calculated
                print('...perform interpolation')
                # split coordinates into components
                coord_UL, coord_LL = np.asarray(Mo_UL), np.asarray(Mo_LL) # transform to numpy array
                x_UL, y_UL, z_UL = coord_UL[:,0], coord_UL[:,1], coord_UL[:,2] # slice list
                x_LL, y_LL, z_LL = coord_LL[:,0], coord_LL[:,1], coord_LL[:,2] # slice list
                # extend coordinates to avoid interpolation artifacts
                x_UL_ext, y_UL_ext, z_UL_ext = ExtendCoordinates(x_UL, y_UL, z_UL, cell)
                x_LL_ext, y_LL_ext, z_LL_ext = ExtendCoordinates(x_LL, y_LL, z_LL, cell)
                # calculate interlayer distance
                x_grid, y_grid, A_REF = MakeGrid(cell, twist_angle) # create interpolation grid
                z_UL_int = DoInterpolation(x_grid, y_grid, x_UL_ext, y_UL_ext, z_UL_ext) # interpolate UL
                z_LL_int = DoInterpolation(x_grid, y_grid, x_LL_ext, y_LL_ext, z_LL_ext) # interpolate LL
                d_local = GetInterlayerDistance(z_UL_int, z_LL_int) # calculate interlayer distance
                # store results in pickle file
                with open(filename+'.pickle', 'wb') as f:
                    pickle.dump([Mo_UL, Mo_LL, cell, x_grid, y_grid, d_local, A_REF], f) # store in pickle file


            ### perform plotting (not separated into functions)
            
            ## prepare parameters: size plotting area
            # window of (x,y) coord to plot relative to origin [Ang]
            X_NEG, X_POS = A_REF, 2*A_REF
            Y_NEG, Y_POS = int(0.5*A_REF), int(1.5*A_REF)
            # get cell constants
            a = np.sqrt(cell[0][0]**2+cell[0][1]**2+cell[0][2]**2) # cell constant a
            b = np.sqrt(cell[1][0]**2+cell[1][1]**2+cell[1][2]**2) # cell constant b
            # find how much to extend in terms of supercells
            x_left = math.ceil((X_NEG + 0.5*a)/a)+2
            x_right = math.ceil(X_POS/a)+1
            y_bottom = math.ceil(Y_NEG/(SIN60*b))+1
            y_top = math.ceil(Y_POS/(SIN60*b))+1
            
            ## prepare interpolated data: extend within supercell of plotting area
            x_sc, y_sc, d_local_sc = [], [], [] # prepare lists
            for m in range(-x_left-1, x_right+1): # supercell in x-direction
                for n in range(-y_bottom-1, y_top+1): # supercell in y-direction
                    for i in range(len(d_local)): # over all data points
                        x_sc.append(x_grid[i]+m*cell[0][0]+n*cell[1][0]) # x coordinates
                        y_sc.append(y_grid[i]+m*cell[0][1]+n*cell[1][1]) # y coordinates
                        d_local_sc.append(d_local[i]) # local interlayer distance
                        
            ## prepare plotting
            print('...plot interlayer distance')
            # initialize plot object
            fig = plt.figure() # initialize figure
            ax = fig.add_subplot(111, aspect='equal') # equal aspect for axes
            # decide parameters for line plotting
            width = 100
            linestyle = 'k-' 
            # get size of length bar depending on A_REF
            if A_REF==20: SIZE_BAR = 10 # 1 nm bar
            elif A_REF==50: SIZE_BAR = 20 # 2 nm bar
            elif A_REF==200: SIZE_BAR = 100 # 10 nm bar
            else: SIZE_BAR = 500 # 50 nm bar
            # get twist angle with correct number of decimals
            if twist_angle>=1 and twist_angle<=59: # two decimal places
                angle_string = r"$\theta=$"+str("{:5.2f}".format(twist_angle) )+'°' # 2 characters before decimal, 2 after
            else: angle_string = r"$\theta=$"+str("{:5.3f}".format(twist_angle) )+'°' # 2 characters before decimal, 3 after
            # define position of length bar in bottom right
            OFFSET_X = 0.30*A_REF # position from right side
            OFFSET_Y = 0.20*A_REF # position from bottom
            WIDTH_Y =  0.04*A_REF # height of vertical lines
            # define position where twist angle is shown on top left
            LABEL_X = -X_NEG + 2.2/50*A_REF 
            LABEL_Y = Y_POS - 7.2/50*A_REF
            
            ## add auxiliary elements to plot
            # size bar in bottom right
            plt.plot([X_POS-OFFSET_X-SIZE_BAR, X_POS-OFFSET_X], [-Y_NEG+OFFSET_Y, -Y_NEG+OFFSET_Y], linestyle, ms=width, zorder=100) # main line
            plt.plot([X_POS-OFFSET_X-SIZE_BAR, X_POS-OFFSET_X-SIZE_BAR], [-Y_NEG+OFFSET_Y+WIDTH_Y, -Y_NEG+OFFSET_Y-WIDTH_Y], linestyle, ms=width, zorder=100) # left border
            plt.plot([X_POS-OFFSET_X, X_POS-OFFSET_X], [-Y_NEG+OFFSET_Y+WIDTH_Y, -Y_NEG+OFFSET_Y-WIDTH_Y], linestyle, ms=width, zorder=100) # right border
            plt.text(X_POS-OFFSET_X-0.5*SIZE_BAR, -Y_NEG+OFFSET_Y+0.5*WIDTH_Y, s=str(int(SIZE_BAR/10))+" nm", fontsize='large', horizontalalignment='center', zorder=100) # add label
            # show twist angle in top left of plot
            plt.text(LABEL_X, LABEL_Y, s=angle_string, fontsize='x-large', horizontalalignment='left', bbox=dict(boxstyle="square", ec=(0., 0., 0.), fc=(1., 1., 1.)), zorder=100) # add label
        
            ## plot interlayer distance
            # if wanted: add auxiliary points to get consistent (fixed) scale of colorbar (there might be a better way, but this works nicely as well)
            if FIXED_COLORBAR:
                x_sc.append(-1000); y_sc.append(-1000); d_local_sc.append(D_MIN) # minimum value
                x_sc.append(-1000); y_sc.append(-1000); d_local_sc.append(D_MAX) # maximum value
            # add scatterplot of interlayer distance
            im = ax.scatter(x_sc, y_sc, c=d_local_sc, s=SIZE_PT) # scatter data
            
            ## finalize plot
            # handle colorbar corresponding to interlayer distance
            divider = make_axes_locatable(ax) # create new axis object for colorbar
            cax = divider.append_axes("right", size="4%", pad=0.25) # 2.5% width of plot, distance to plot of pad
            np.arange(round(D_MIN,1), D_MAX+1E-6, 0.1)
            cb = fig.colorbar(im, cax=cax, ticks=np.arange(round(D_MIN,1), D_MAX+1E-6, 0.1)) # create colorbar
            cb.set_label(label="d [Å]") # label colorbar
            cb.ax.tick_params() # set tick label size of colorbar
            # set limits
            ax.set_xlim(-X_NEG, X_POS) # set x limits
            ax.set_ylim(-Y_NEG, Y_POS) # set y limits
            # remove ticks for x and y axis
            ax.get_xaxis().set_ticks([]) # don't show x-axis ticks
            ax.get_yaxis().set_ticks([]) # don't show y-axis ticks
            # tight layout
            plt.tight_layout() # avoid problems with cut off labels
            # save plot and finish
            plt.savefig(filename+'.png', dpi=DPI)
            plt.close()
            

#===========#
# Functions #
#===========#

### read xsf (XCrySDen) file, assuming a standard file layout
def ReadXsf(filename):
    '''
    Import simulation cell and atomic coordinates from xsf file

    Parameters
    ----------
    filename : string
        Filename of structure without extension.

    Returns
    -------
    cell : 2D list
        Simulation cell vectors: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]].
    atoms : 2D list
        Atom types (by atom number) and coordinates: [atoms][atomic number, x, y, z].
    '''
    print("...reading xsf file")
    ## open file, read lines
    try: 
        file = open(filename + '.xsf')
        lines = file.readlines()
        file.close()
    except: 
        sys.exit('Error code 01: Input file ' + filename + '.xsf could not be opened.')
    # assumed .xsf file layout, giving the first few lines
    '''
    Assumed format:
        CRYSTAL
        PRIMVEC
        a_x   a_y   a_z
        b_x   b_y   b_z
        c_x   c_y   c_z
        PRIMCOORD
        Z 1
        Atomic_number   x_i   y_i   z_i
    With:
        * keywords in uppercase
        * a_i, b_i, and c_i is the simulation cell
        * Z is the number of atoms
        * atoms are given in the end with their atomic number and x,y,z coordinates (first line is shown)
    '''
    ## check if .xsf file follows assumed format
    line_crystal, line_primvec, line_primcoord = None, None, None # lines in which keywords are found
    for i in range(15): # check all lines
    # for i in range(len(lines)): # check all lines
        if "CRYSTAL" in lines[i]: line_crystal = i # keyword for 3D periodic system
        if "PRIMVEC" in lines[i]: line_primvec = i # keyword for simulation cell vectors
        if "PRIMCOORD" in lines[i]: line_primcoord = i # keyword for coordinates
    ## if problem with file: stop iteration if keywords were not found
    if line_crystal==None: sys.exit('Error code 02: xsf file needs to given as 3D system (CRYSTAL)!')
    if line_primvec==None: sys.exit('Error code 03: xsf file needs primitive vectors (PRIMVEC) or has keyword!')
    if line_primcoord==None: sys.exit('Error code 04: xsf file needs primitive coordinates (PRIMCOORD)!')
    ## read unit cell
    cell = [] # create empty list for cell parameters
    vec_a, vec_b, vec_c = lines[line_primvec+1].split(), lines[line_primvec+2].split(), lines[line_primvec+3].split() # read vectors from file
    cell.append([float(vec_a[0]), float(vec_a[1]), float(vec_a[2])]) # write vector a to array
    cell.append([float(vec_b[0]), float(vec_b[1]), float(vec_b[2])]) # write vector b to array
    cell.append([float(vec_c[0]), float(vec_c[1]), float(vec_c[2])]) # write vector c to array
    ## read number of atoms
    line_Z = lines[line_primcoord+1].split() # find correct line
    Z = int(line_Z[0]) # atom number Z is first value in line
    ## read atom types and coordinates 
    atoms = [] # create array for output
    for i in range(line_primcoord+2, line_primcoord+2+Z): # iterate over all lines with coordinates
        line_i = lines[i].split() # read lines, split into elements
        atoms.append([int(line_i[0]), float(line_i[1]), float(line_i[2]), float(line_i[3])]) # [atom number, x, y, z]
    ## return statement
    return cell, atoms
        
      
### sort Mo atoms into UL and LL
def SortAtoms(cell, atoms):
    '''
    Sort Mo atoms into layers.

    Parameters
    ----------
    cell : 2D list
        Simulation cell vectors: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]].
    atoms : 2D list
        Atom types (by atom number) and coordinates: [atoms][atomic number, x, y, z].

    Returns
    -------
    Mo_UL: 2D list
        Coordinates in upper layer: [atoms][x, y, z].
    Mo_LL: 2D list
        Coordinates in lower layer: [atoms][x, y, z].
    '''
    print("...sorting atoms")
    ## extract the Mo coordinates
    coord_Mo = []
    for atom in atoms: # over all atoms
        if atom[0]==42: coord_Mo.append([atom[1], atom[2], atom[3]]) # Mo atom: extract coordinates
        elif atom[0]==16: continue # S atom: not use further
        else: sys.exit('Error code 05: invalid atom type encountered! Only Mo (42) and S (16) are expected.')
    ## enforce periodic boundary conditions (PBC)
    A = np.array([[cell[0][0], cell[1][0], cell[2][0]], [cell[0][1], cell[1][1], cell[2][1]], [cell[0][2], cell[1][2], cell[2][2]]]) # transformation matrix
    for i in range(len(coord_Mo)): # iterate over all atoms
        # calculate fractional coordinates
        [x, y, z] = np.linalg.solve(A, np.asarray(coord_Mo[i]))
        # enforce PBC
        x, y, z = x%1, y%1, z%1
        # transform back to absolute coordinates
        x_prim = x*cell[0][0] + y*cell[1][0] + z*cell[2][0]
        y_prim = x*cell[0][1] + y*cell[1][1] + z*cell[2][1]
        z_prim = x*cell[0][2] + y*cell[1][2] + z*cell[2][2]
        # overwrite original values
        coord_Mo[i] = [x_prim, y_prim, z_prim]
    ## normal sorting: below vs. above center of mass in z-direction
    if not STRONG_CORRUGATION:
        # sort Mo atoms by center of mass in z-direction
        Mo_LL, Mo_UL, z_sum = [], [], 0 # coordinates LL, UL and sum of z coordinates for center of mass calculation
        for i in range(len(coord_Mo)): z_sum += coord_Mo[i][2] # sum z coordinates
        z_mean = z_sum / len(coord_Mo) # average coordinate
        for i in range(len(coord_Mo)): # check all Mo atoms
            if coord_Mo[i][2]<z_mean: Mo_LL.append(coord_Mo[i]) # lower layer
            else: Mo_UL.append(coord_Mo[i]) # upper layer
        # sanity check: correct stoichiometry?
        if len(atoms)!=3*len(coord_Mo): # MoS2 stoichiometry 
            print('   Structure does not follow MoS2 stoichiometry! If this is wanted (e.g. strained systems) please comment out the next line which stops the script.')
            sys.exit('Error code 06: structure file has wrong stoichiometry!')
        # sanity check: lower and upper layer same number of atoms?
        if len(Mo_LL)!=len(Mo_UL):
            print('   If this is wanted (e.g. strained systems) please comment out the next line stopping the script!')
            sys.exit('Error code 07: different number of atoms in lower and upper layer!')
    ## extended sorting: search by neighbors -> needed only when layers are too strongly corrugated
    # when needed: when layers show a z-coordinate variation reaching beyond the mean z coordinate
    # how to spot: interlayer distance landscape will show artifacts instead of a slow variation
    if STRONG_CORRUGATION:
        Mo_LL, Mo_UL = [], [] # initialize coordinates LL and UL
        # local function: give distance between two atoms
        def Distance(a,b):
            '''
            Input: np.array x,y,z coordinates
            Output: distance between points a and b
            '''
            return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
        # create list of only the Mo atom z coordinates
        z_Mo = []
        for i in range(len(coord_Mo)): z_Mo.append(coord_Mo[i][2])
        # search for highest and lowest z as seeds for layer search
        index_min, index_max = np.argmin(z_Mo), np.argmax(z_Mo)
        # prepare lists for search routine
        coord_Mo = np.asarray(coord_Mo) # make numpy array for easier computation
        LL = np.array([index_min]) # atomic indizes for lower layer
        UL = np.array([index_max]) # atomic indizes for upper layer
        indizes = np.asarray(range(len(coord_Mo))) # all atom indizes stored for sorting routine
        # remove already found indizes from index array
        indizes = np.setdiff1d(indizes,LL) # remove lower layer indizes
        indizes = np.setdiff1d(indizes,UL) # remove upper layer indizes
        # new index list UL and LL for first step
        LL_new = np.copy(LL)
        UL_new = np.copy(UL)
        # iterate over indexes until all atoms are sorted
        while len(indizes)>0: # iterate as long as there are atoms left to sort
            # search for nearest neighbors of UL_new in indizes
            UL_new2 = np.array([]) # create list for results
            for i in UL_new: # indizes of seeds in UL
                for k in indizes: # remaining indexes
                    # if neighbor: add index to newly found neighbors list
                    if Distance(coord_Mo[int(i)],coord_Mo[int(k)])<=D_Mo_THRESH: UL_new2 = np.append(UL_new2, int(k)) 
            # search for nearest neighbors of LL_new in indizes
            LL_new2 = np.array([]) # create list for results
            for i in LL_new: # indizes of seeds in LL
                for k in indizes: # remaining indexes
                    # if neighbor: add index to newly found neighbors list
                    if Distance(coord_Mo[int(i)],coord_Mo[int(k)])<=D_Mo_THRESH: LL_new2 = np.append(LL_new2, int(k)) 
            # remove dublicates from newly found indizes to avoid problems
            UL_new2 = np.unique(UL_new2); LL_new2 = np.unique(LL_new2)
            # add newly found indizes to list of known indexes
            UL = np.append(UL, UL_new2); LL = np.append(LL, LL_new2)
            # remove newly found indizes from remaining indizes
            indizes = np.setdiff1d(indizes, UL_new2); indizes = np.setdiff1d(indizes, LL_new2)
            # set new2-arrays to new-arrays for next iteration
            UL_new = np.copy(UL_new2); LL_new = np.copy(LL_new2)  
        # from results create final layer arrays: sort results arrays to get proper coordinates
        UL = np.unique(UL); LL = np.unique(LL)
        # write coordinates to output lists
        for i in UL: Mo_UL.append(coord_Mo[int(i)])
        for i in LL: Mo_LL.append(coord_Mo[int(i)])
    ## return statement
    return Mo_UL, Mo_LL


### extend coordinates around simulation cell
def ExtendCoordinates(x, y, z, cell):
    '''
    Extend coordinates around simulation cell to avoid artifacts during interpolation

    Parameters
    ----------
    x, y, z : list
        Atomic coordinates within a single layer, split into x-, y- and z-component.
    cell : 2D list
        Simulation cell vectors: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]].

    Returns
    -------
    x_sc, y_sc, z_sc : list
        Extended coordinates, split into x-, y- and z-component.
    '''
    ## preparation
    # lists for new values
    x_sc, y_sc, z_sc = [], [], [] # create new lists
    x_sc_frac, y_sc_frac = [], [] # create new lists
    # transformation matrix to obtain fractional coordinates
    A = np.array([[cell[0][0], cell[1][0]], [cell[0][1], cell[1][1]]])
    # calculate what fractional value DIST_EXTEND corresponds to
    FRAC_CUTOFF = DIST_EXTEND / np.sqrt(cell[0][0]**2+cell[0][1]**2+cell[0][2]**2)
    ## make 3x3 supercell from initial coordinates
    for i in range(len(x)): # over all atoms
        # calculate fractional coordinates
        vec_x = np.array([x[i], y[i]]) # coordinate vector
        [x_frac, y_frac] = np.linalg.solve(A, vec_x) # get fractional coordinates
        # apply supercell
        for m in [-1,0,1]: # supercell in x-direction
            for n in [-1,0,1]: # supercell in y-direction
                x_test = x_frac + m
                y_test = y_frac + n
                # filter out what needed
                if x_test>=-FRAC_CUTOFF and x_test<=(1+FRAC_CUTOFF):
                    if y_test>=-FRAC_CUTOFF and y_test<=(1+FRAC_CUTOFF):
                        x_sc_frac.append(x_test)
                        y_sc_frac.append(y_test)
                        z_sc.append(z[i])
    ## go back from fractional to cartesian coordinates
    for i in range(len(x_sc_frac)): # over all coordinates
        x_sc.append(x_sc_frac[i]*cell[0][0]+y_sc_frac[i]*cell[1][0])
        y_sc.append(x_sc_frac[i]*cell[0][1]+y_sc_frac[i]*cell[1][1])  
    ## return statement
    return x_sc, y_sc, z_sc


### create interpolation grid inside simulation cell
def MakeGrid(cell, theta):
    '''
    Set up grid for interpolating the simulation cell (based on a lot of trial and error to make look nice).

    Parameters
    ----------
    cell : 2D list
        Simulation cell vectors: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]].
    theta : float
        Twist angle in degree.

    Returns
    -------
    x_grid, y_grid : list
        Interpolation grid with separate x- and y-coordinates.
    A_REF : integer
        Reference value for size of moiré cell, needed for plotting.
    '''
    ## calculate analytical moiré cell
    if theta>30: theta = 60 - theta # transform to 60°-theta
    a_moire = A_ML / np.sqrt(2*(1-np.cos(np.deg2rad(theta)))) # moiré cell constant approximated by monolayer (does not include compression in tBL)
    ## identify a suitable plotting reference depending on moiré cell size (from trial and error)
    if a_moire<=20: A_REF = 20 # tiny cell size
    elif a_moire>20 and a_moire<=50: A_REF = 50 # small cell size
    elif a_moire>50 and a_moire<=200: A_REF = 200 # intermediate size
    else: A_REF = 1000 # large >200 Ang
    ## set up grid steps from cell size
    a = np.sqrt(cell[0][0]**2 + cell[0][1]**2 + cell[0][2]**2) # cell constant
    GRID_STEPS = int(77.832*(a/A_REF)+9.8418)+1 # linear function fitted to results from previous tests what looks nice without artifacts
    ## create grid
    x_grid, y_grid = [], [] # new lists
    for i in range(0, GRID_STEPS): # along a direction
        for k in range(0, GRID_STEPS): # along b direction
            x_grid.append(i*cell[0][0]/GRID_STEPS + k*cell[1][0]/GRID_STEPS)
            y_grid.append(i*cell[0][1]/GRID_STEPS + k*cell[1][1]/GRID_STEPS)
    ## return statement
    return x_grid, y_grid, A_REF


### perform interpolation of z-coordinates
def DoInterpolation(x_grid, y_grid, x_ext, y_ext, z_ext):
    '''
    Perform interpolation of extended coordinates on a given grid

    Parameters
    ----------
    x_grid, y_grid : list
        Interpolation grid with separate x- and y-coordinates.
    x_ext, y_ext, z_ext : list
        Coordinates, extended around simulation cell, separated x-, y-, and z-components.

    Returns
    -------
    z_int : list
        Interpolated z-coordinates, corresponding to grid points.
    '''
    ## interpolate using griddata method from SciPy
    z_int = interpolate.griddata(points=(x_ext, y_ext), values=z_ext, xi=(x_grid, y_grid), method=GRIDDATA_ORDER)
    ## return statement
    return z_int


### calculate local interlayer distance
def GetInterlayerDistance(z_UL, z_LL):
    '''
    Calculate interlayer distance from interpolated coordinates

    Parameters
    ----------
    z_UL, z_LL : list
        Interpolated z-coordinates, corresponding to grid points.

    Returns
    -------
    d_local : list
        Local interlayer distance at grid points.
    '''
    ## calculate property
    d_local = []
    for i in range(len(z_UL)): # for all grid points
        d_local.append(z_UL[i] - z_LL[i]) # d = z(UL) - z(LL)
    ## return statement
    return d_local
        

#============#
# Run script #
#============#
    
if __name__ == '__main__':
    main()   