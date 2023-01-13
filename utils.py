import numpy as np

def gridmap_generator(height,widht,cell_size):
    cs = cell_size
    cell_x    = int(height/cs)
    cell_y    = (int(widht/cs))*cell_x
    gridmap = [np.array([[0,0],[cs,0],[cs,cs],[0,cs]])]
    # Built the first row of the gridmap
    for i in range(cell_x-1):
        new_cell = sum([gridmap[i],np.array([[cs,0],[cs,0],[cs,0],[cs,0]])])
        gridmap.append(new_cell)
    #Build the next rows until completing the gridmap
    for i in range(cell_y-cell_x):
        new_cell_r = sum([gridmap[i],np.array([[0,cs],[0,cs],[0,cs],[0,cs]])])
        gridmap.append(new_cell_r)

    return gridmap

def p2l(p):
    return np.log(p/(1-p))