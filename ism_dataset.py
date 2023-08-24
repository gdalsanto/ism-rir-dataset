import os
import csv
import time
import uuid
import argparse
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

import pyroomacoustics as pra

np.random.seed(0)

# meatrial array 
# walls 
wall_mat = {
"plasterboard" : {
    "description": "2 * 13 mm plasterboard on steel frame, 50 mm mineral wool in cavity, surface painted",
    "coeffs": [0.15, 0.10, 0.06, 0.04, 0.04, 0.05, 0.05], 
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
},
"wooden_lining" : {
    "description": "Wooden lining, 12 mm fixed on frame",
    "coeffs": [0.27, 0.23, 0.22, 0.15, 0.10, 0.07, 0.06],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
},
"limestone_wall" : {
    "description": "Limestone walls",
    "coeffs": [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
},
"smooth_brickwork_10mm_pointing": {
    "description": "Smooth brickwork, 10 mm deep pointing, pit sand mortar",
    "coeffs": [0.08, 0.09, 0.12, 0.16, 0.22, 0.24, 0.24],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
}
}
# floor
floor_mat = {
"audience_floor" : {
    "description": "Audience floor, 2 layers, 33 mm on sleepers over concrete",
    "coeffs": [0.09, 0.06, 0.05, 0.05, 0.05, 0.04],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
},
"carpet_cotton" : {
        "description": "Cotton carpet",
        "coeffs": [0.07, 0.31, 0.49, 0.81, 0.66, 0.54, 0.48],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
},
"carpet_hairy": {
    "description": "Hairy carpet on 3 mm felt",
    "coeffs": [0.11, 0.14, 0.37, 0.43, 0.27, 0.25, 0.25],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
},
"carpet_tufted_9m" : {
    "description": "9 mm tufted pile carpet on felt underlay",
    "coeffs": [0.08, 0.08, 0.30, 0.60, 0.75, 0.80, 0.80],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
}
}
# ceiling
ceiling_mat = {
"ceiling_plasterboard" : {
    "description": "Plasterboard ceiling on battens with large air-space above",
    "coeffs": [0.20, 0.15, 0.10, 0.08, 0.04, 0.02],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
},
"ceiling_melamine_foam" : {
    "description": "Wedge-shaped, melamine foam, ceiling tile",
    "coeffs": [0.12, 0.33, 0.83, 0.97, 0.98, 0.95],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
},
"ceiling_metal_panel": {
    "description": "Metal panel ceiling, backed by 20 mm Sillan acoustic tiles, panel width 85 mm, panel spacing 15 mm, cavity 35 cm",
    "coeffs": [0.59, 0.80, 0.82, 0.65, 0.27, 0.23],
    "center_freqs": [125, 250, 500, 1000, 2000, 4000],
}
}

# room dimension
room_dim_opt = {
    "small": {"length": [2, 7],"width": [1.5, 5], "height": [2.4, 3]},
    "medium": {"length": [7, 15],"width": [5, 10], "height": [3, 5]},
    "large": {"length": [15, 25],"width": [10, 20], "height": [5, 10]}
}

# Create the shoebox
# material: (absorption, scattering)
materials_mu = {
    'ceiling':(0.25, 0.01),
    'floor':(0.5, 0.1),
    'east':(0.15, 0.15),
    'west':(0.07, 0.15),
    'north':(0.15, 0.15),
    'south':(0.10, 0.15),
}

def sample_room_dim(dim_limits):
    room_l = np.random.uniform(dim_limits['length'][0], dim_limits['length'][1])
    room_w = np.random.uniform(dim_limits['width'][0], dim_limits['width'][1])
    room_h = np.random.uniform(dim_limits['height'][0], dim_limits['height'][1])
    room_dim = [room_l, room_w, room_h]
    return np.round(np.multiply(room_dim,1000))/1000

def sample_room_materials():
    s = np.random.uniform(0.0, 0.2)
    a1 = np.random.uniform(0.0, 1)
    a2 = np.random.uniform(0.0, 1)
    a3 = np.random.uniform(0.0, 1)

    materials = pra.make_materials(
        ceiling=(a1, s),
        floor=(a2, s),
        east=(a3, s),
        west=(a1, s),
        north=(a2, s),
        south=(a3, s),
    )  
    return materials, [s, a1, a2, a3]

def sample_materials_from_dict():
    wall_n = np.random.randint(0, len(wall_mat.keys()))
    floor_n = np.random.randint(0, len(floor_mat.keys()))
    ceiling_n = np.random.randint(0, len(ceiling_mat.keys()))
    # get materials 
    cm = list(ceiling_mat.keys())[ceiling_n]
    fm = list(floor_mat.keys())[floor_n]
    wm = list(wall_mat.keys())[wall_n]
    materials = pra.make_materials(
        ceiling=ceiling_mat[cm],
        floor=floor_mat[fm],
        east=wall_mat[wm],
        west=wall_mat[wm],
        north=wall_mat[wm],
        south=wall_mat[wm],
    )    
    return materials, [cm, fm, wm]
    
def sample_room_interior(room_dim):
    x = np.random.uniform(0.01, room_dim[0]-0.01)
    y = np.random.uniform(0.01, room_dim[1]-0.01)
    z = np.random.uniform(0.01, room_dim[2]-0.01)
    coord = [x, y, z]
    return np.round(np.multiply(coord,1000))/1000


def make_room(config):
    """
    A short helper function to make the room according to config
    """

    shoebox = (
        pra.ShoeBox(
            config['room_dim'],
            materials=config['materials'],
            # materials=pra.Material.make_freq_flat(0.07),
            fs=config['sr'],
            max_order=config["max_order"],
            ray_tracing=config["ray_tracing"],
            air_absorption=True,
        )
        .add_source(config['source'])   
        .add_microphone(config['mic'])
    )

    return shoebox


def chrono(f, *args, **kwargs):
    """
    A short helper function to measure running time
    """
    t = time.perf_counter()
    ret = f(*args, **kwargs)
    runtime = time.perf_counter() - t
    return runtime, ret


def main(args):
    
    config = {}
    config['max_order'] = 17
    config['ray_tracing'] = True
    config['sr'] = args.sr
    
    # get room sizes splits 
    splits = np.floor(np.cumsum(
        np.divide(args.split, sum(args.split)) * args.n_rir))

    # open metadata file 
    filepath = os.path.join(args.dir_path, 'metadata.csv')
    fieldnames = ['filename', 'size', 'room_dim', 'material_ceil', 'material_floor', 'material_wall', 'source', 'mic', 'rt60', 'max_order', 'ray_tracing', 'sr']
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()        
        for i_rir in tqdm(range(args.n_rir)):
            # detect room size 
            if i_rir < splits[0]:
                room_type = 'small'
            elif i_rir < splits[1]:
                room_type = 'medium'
            else: 
                room_type = 'large'

            # sample the shoebox room configuration parameters 
            # dimension
            config['room_dim'] = sample_room_dim(room_dim_opt[room_type])
            config['materials'], materials = sample_materials_from_dict()
            config['source'] = sample_room_interior(config['room_dim'])
            config['mic'] = sample_room_interior(config['room_dim'])
            
            shoebox = make_room(config)

            # rt60_sabine = shoebox.rt60_theory(formula="sabine")
            # rt60_eyring = shoebox.rt60_theory(formula="eyring")
            # compute rir
            shoebox.image_source_model()
            shoebox.ray_tracing()
            shoebox.compute_rir()
            rir = shoebox.rir[0][0].copy()
            config['rt60'] = shoebox.measure_rt60(decay_db=60)
            config['material_ceil'] = materials[0] 
            config['material_floor'] = materials[1]
            config['material_wall'] = materials[2]
            config['size'] = room_type
            del config['materials']

            # save file 
            config['filename'] = os.path.join(args.dir_path, str(uuid.uuid4()) + '.wav')
            sf.write(config['filename'], rir/np.max(np.abs(rir)), shoebox.fs)
            writer.writerow(config)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_rir', default = 200000, type=int,
        help = 'Number of RIRs to be simulated')
    parser.add_argument('--sr', default = 48000, type=int,
        help='Sampling rate')
    parser.add_argument('--split',  nargs='+', type=float,
        help='room size contribution in parts - samll, medium, large')
    parser.add_argument('--dir_path', 
    help='path to directory where to save the dataset')

    args = parser.parse_args()

    main(args)

