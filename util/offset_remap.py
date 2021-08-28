import torch
import numpy as np 

def test_offset_map():
    a = np.linspace(1,9,9).reshape(3,3)
    b = np.array([[0,1,2],[0,1,2]])
    a_tensor = torch.from_numpy(a)
    b_tensor = torch.from_numpy(b)
    zeros = torch.zeros_like(a_tensor)
    np_zeros = np.zeros_like(a)
    np_zeros[[2],[1]] = a
    # zeros[b_tensor] = a_tensor
    print(np_zeros)
    print(zeros)

def img_remap_coordinates(input,coords):
    input_height = input.shape[0]
    input_width = input.shape[1]
    input_channel = input.shape[2]
    n_coords = coords.shape[0]

    coords = np.stack( [np.clip(coords[:,0],0,input_height-1), np.clip(coords[:,1],0,input_width-1)],axis=-1)
    index = coords[:,0]*input_height + coords[:,1]
    
    input = input.reshape([-1,input_channel])
    zeros = np.zeros_like(input)
    mask = np.zeros_like(input)
    ones = np.ones_like(input)
    zeros[index] = input
    mask[index] = ones
    remap = zeros.reshape([input_height,input_width,input_channel])
    mask = mask.reshape([input_height,input_width,input_channel])
    return remap,mask

def generate_grid(height,width):
    grid = np.meshgrid(range(height),range(width),indexing='ij')
    grid = np.stack(grid,axis=-1)
    grid = grid.reshape(-1,2)
    return grid

def img_offset_remap(offset, img_offset):


    input_height = img_offset.shape[0]
    input_width = img_offset.shape[1]

    offset = offset.astype(np.int)
    offset = offset.reshape([-1,2])
    grid = generate_grid(input_height,input_width)

    coords = grid + offset
    remap, mask = img_remap_coordinates(img_offset,coords)
    return remap, mask