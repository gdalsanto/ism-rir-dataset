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

# general scattering
scattering = {
    "diffuser_ceil": {
    "description": "random diffuser",
    "coeffs": [0.01, 0.01, 0.01, 0.01, 0.01],
    "center_freqs": [125, 250, 500, 1000, 2000],
},
    "diffuser_wall": {
    "description": "random diffuser",
    "coeffs": [0.01, 0.08, 0.15, 0.25, 0.5],
    "center_freqs": [125, 250, 500, 1000, 2000],
},
    "diffuser_floor": {
    "description": "random diffuser",
    "coeffs": [0.01, 0.10, 0.15, 0.25, 0.5],
    "center_freqs": [125, 250, 500, 1000, 2000],
}   
}

# room dimension
room_dim_opt = {
    "small": {  "length": [1.5, 7], "width": [1.5, 5],  "height": [2, 3]},
    "medium": { "length": [5, 10],  "width": [3, 8],    "height": [2, 5]},
    "large": {  "length": [7, 15],  "width": [5, 12],   "height": [3, 7]}
}


'''
# material: (absorption, scattering)
materials_mu = {
    'ceiling':(0.25, 0.01),
    'floor':(0.5, 0.1),
    'east':(0.15, 0.15),
    'west':(0.07, 0.15),
    'north':(0.15, 0.15),
    'south':(0.10, 0.15),
}
'''

def sample_room_dim(dim_limits):
    room_l = np.random.uniform(dim_limits['length'][0], dim_limits['length'][1])
    room_w = np.random.uniform(dim_limits['width'][0], dim_limits['width'][1])
    room_h = np.random.uniform(dim_limits['height'][0], dim_limits['height'][1])
    room_dim = [room_l, room_w, room_h]
    return np.round(np.multiply(room_dim,1000))/1000

def sample_room_materials():
    s = np.random.uniform(0.0, 0.2)
    a1 = np.random.uniform(0.0, 0.5)
    a2 = np.random.uniform(0.0, 0.5)
    a3 = np.random.uniform(0.0, 0.5)

    materials = pra.make_materials(
        ceiling=(a1, s),
        floor=(a2, s),
        east=(a3, s),
        west=(a1, s),
        north=(a2, s),
        south=(a3, s),
    )  
    return materials, [s, a1, a2, a3]

def add_noise_materials(material, p_coeff=np.array(0.1), p_freq=np.array(0.1)):

    coeff = np.array(material['coeffs'])
    material['coeffs'] = (coeff + np.random.uniform(coeff*(-p_coeff), coeff*(p_coeff))).tolist()
    freq = np.array(material['center_freqs'])
    material['center_freqs'] = (freq + np.random.uniform(freq*(-p_freq), freq*(p_freq))).tolist()
    return material 

def sample_materials_from_dict():
    wall_n = np.random.randint(0, len(wall_mat.keys()))
    floor_n = np.random.randint(0, len(floor_mat.keys()))
    ceiling_n = np.random.randint(0, len(ceiling_mat.keys()))
    # get materials 
    cm = list(ceiling_mat.keys())[ceiling_n]
    fm = list(floor_mat.keys())[floor_n]
    wm = list(wall_mat.keys())[wall_n]

    # add noise 
    ceiling = add_noise_materials(  ceiling_mat[cm], 
                                    p_freq = np.logspace(np.log10(0.05), np.log10(0.25), len(ceiling_mat[cm]['center_freqs'])))
    floor = add_noise_materials(floor_mat[fm], 
                                p_freq = np.logspace(np.log10(0.05), np.log10(0.25), len(floor_mat[fm]['center_freqs'])))
    wall = add_noise_materials( wall_mat[wm], 
                                p_freq = np.logspace(np.log10(0.05), np.log10(0.25), len(wall_mat[wm]['center_freqs'])))
    materials = pra.make_materials(
        ceiling=ceiling,
        floor=floor,
        east=wall,
        west=wall,
        north=wall,
        south=wall,
    )    
    '''
    materials = pra.make_materials(
        ceiling=(ceiling_mat[cm], scattering['diffuser_ceil']),
        floor=(floor_mat[fm], scattering['diffuser_ceil']),
        east=(wall_mat[wm], scattering['diffuser_ceil']),
        west=(wall_mat[wm], scattering['diffuser_ceil']),
        north=(wall_mat[wm], scattering['diffuser_ceil']),
        south=(wall_mat[wm], scattering['diffuser_ceil']),
    )    
    '''
    return materials, [cm, fm, wm]
    
def sample_room_interior(room_dim):
    '''
    sample coordinates of a point inside the room
    '''
    x = np.random.uniform(0.25, room_dim[0]-0.25)
    y = np.random.uniform(0.25, room_dim[1]-0.25)
    z = np.random.uniform(0.25, room_dim[2]-0.25)
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
            use_rand_ism = True,
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
    config['max_order'] = args.max_order
    config['ray_tracing'] = args.ray_tracing
    config['sr'] = args.sr
    
    # get room sizes splits 
    splits = np.floor(np.cumsum(
        np.divide(args.split, sum(args.split)) * args.n_rir))

    # open metadata file 
    if not os.path.exists(args.dir_path):
        os.makedirs(args.dir_path)
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

            while config['rt60'] >= args.max_rt60:
                # repeat the above untill you get a rt60 within the acceptable range
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
            csvfile.flush()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_rir', default = 100, type=int,
        help = 'Number of RIRs to be simulated')
    parser.add_argument('--sr', default = 48000, type=int,
        help='Sampling rate')
    parser.add_argument('--split',  nargs='+', type=float,
        help='room size contribution in parts - samll, medium, large')
    parser.add_argument('--dir_path', 
        help='path to directory where to save the dataset')
    parser.add_argument('--max_rt60', type=float,
        help='Max accepted rt60 value')
    parser.add_argument('--ray_tracing', action='store_true', 
        help='If true hybrid ISM/ray_tracing method will be used')
    parser.add_argument('--max_order', default=100, type=int,
        help='Max order to use in ISM synthesis')
    args = parser.parse_args()

    main(args)

