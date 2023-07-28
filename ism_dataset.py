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
    s = np.random.uniform(0.0, 0.5)
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
    fieldnames = ['filename', 'size', 'room_dim', 'materials', 'source', 'mic', 'rt60', 'max_order', 'ray_tracing', 'sr']
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
            config['materials'], materials = sample_room_materials()
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
            config['materials'] = np.round(np.multiply(materials,1000))/1000 # overwrite for the csv file
            config['size'] = room_type

            # save file 
            config['filename'] = os.path.join(args.dir_path, str(uuid.uuid4()) + '.wav')
            sf.write(config['filename'], rir/np.max(np.abs(rir)), shoebox.fs)
            writer.writerow(config)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_rir', default = 200000, type=int,
        help = 'Number of RIRs to be simulated')
    parser.add_argument('--sr', default = 44100, type=int,
        help='Sampling rate')
    parser.add_argument('--split',  nargs='+', type=float,
        help='room size contribution in parts - samll, medium, large')
    parser.add_argument('--dir_path', 
    help='path to directory where to save the dataset')

    args = parser.parse_args()

    main(args)

