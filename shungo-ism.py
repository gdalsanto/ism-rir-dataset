import numpy as np
import soundfile as sf
import pyroomacoustics as pra
import random
import argparse
import os 
import csv
from tqdm import tqdm
import uuid

def get_random_ir(
        sr=48000, 
        eps=1e-5, 
        maximum_order=100, 
        render_sec=2.5, 
        room_size_base=3, 
        room_size_pow=3,
        smoothing_range=(3, 5),
        smoothing_compensate=True,
        ): 

    x = room_size_base ** np.random.uniform(1, room_size_pow)
    y = room_size_base ** np.random.uniform(1, room_size_pow)
    z = room_size_base ** np.random.uniform(1, room_size_pow)
    
    room_dim = np.array([x, y, z])

    V = np.prod(room_dim)
    S1, S2, S3 = x * y, y * z, z * x

    materials = random_materials(
            sr=sr, 
            smoothing_range=smoothing_range, 
            smoothing_compensate=smoothing_compensate
            )

    S_freqs = (materials['ceiling'].absorption_coeffs + materials['floor'].absorption_coeffs) * S1 \
               + (materials['east'].absorption_coeffs + materials['west'].absorption_coeffs) * S2 \
               + (materials['north'].absorption_coeffs + materials['south'].absorption_coeffs) * S3

    S_min = np.min(S_freqs)

    RT60 = .16 * V / S_min
    RT60 = min(render_sec, RT60)

    temperature = np.random.uniform(-40, 45)
    humidity = np.random.uniform(0, 100)

    c = 331.4 + 0.6 * temperature + 0.0124 * humidity

    max_dist = RT60 * c
    max_order_point = max_dist / np.sqrt((1 / x) ** 2 + (1 / y) ** 2 + (1 / z) ** 2) * np.array([1 / x, 1 / y, 1 / z])
    
    max_order = int(max_order_point[0] / x) + int(max_order_point[1] / y) + int(max_order_point[2] / z)
    #print("actual max order", max_order)
    max_order = min(max_order, maximum_order)

    room = pra.ShoeBox(
        room_dim,
        fs = sr,
        materials = materials,
        max_order = max_order,
        air_absorption = True,
        temperature = temperature,
        humidity = humidity,
        sigma2_awgn = 200
    )

    source_loc = [np.random.uniform(eps, x - eps),
                     np.random.uniform(eps, y - eps),
                     np.random.uniform(eps, z - eps)]

    room.add_source(source_loc,
                    signal = np.array([1]),
                    delay = 0)

    mic_locs = np.c_[[np.random.uniform(eps, x - eps),
                      np.random.uniform(eps, y - eps),
                      np.random.uniform(eps, z - eps)]]

    room.add_microphone_array(mic_locs)
    room.compute_rir()

    return room, mic_locs, source_loc

def random_materials(
        sr=48000, 
        freq_dev=2**(1/6),
        **absorption_kwargs
        ):
    materials = {}

    center_freqs = np.array([15.625, 19.686, 24.803, 31.250, 39.373, 49.606, 62.500, 78.745, 99.213,
                             125.000, 157.490, 198.425, 250.000, 314.980, 396.850, 500.000, 629.961, 
                             793.701, 1000.000, 1259.921, 1587.401, 2000.000, 2519.842, 3174.802, 
                             4000.000, 5039.684, 6349.604, 8000.000, 10079.368, 12699.208, 16000.000, 
                             20158.737])

    dev = freq_dev ** np.random.normal(0, 1, size = (len(center_freqs),))
    center_freqs = center_freqs * dev
    center_freqs = np.sort(center_freqs)
    center_freqs = center_freqs[center_freqs < sr / 2]

    num_bands = len(center_freqs)

    center_absorption = 0.1 ** np.random.uniform(0, 1) # .1, .95
    allowed_deviation = 20 ** np.random.uniform(0, 1)

    num_unique_materials = random.randrange(1, 5)
    wall_materials = []
    for i in range(num_unique_materials):
        wall_materials.append(
                random_absorption(
                    center_absorption, 
                    allowed_deviation, 
                    num_bands, 
                    **absorption_kwargs
                    )
                )

    while len(wall_materials) < 6:
        wall_materials.append(wall_materials[-1])
        random.shuffle(wall_materials)

    i = 0
    for wall in ['ceiling', 'floor', 'east', 'west', 'north', 'south']:
        materials[wall] = (pra.parameters.Material(dict(description = wall,
                                                     coeffs = wall_materials[i],
                                                     center_freqs = center_freqs)))
        i += 1
    return materials

def random_absorption(
        center_absorption, 
        allowed_deviation, 
        num_bands, 
        smoothing_range=(3, 5), 
        smoothing_compensate=True
        ):
    exps = np.random.uniform(-1, 1, size = (num_bands,))
    smoothing = random.randrange(*smoothing_range)
    smoother = np.hanning(smoothing * 2 + 1)
    exps = np.convolve(exps, smoother)[smoothing:-smoothing] 
    if smoothing_compensate:
        exps = exps * np.sqrt(smoothing * 2 - 1)
    absorptions = center_absorption * allowed_deviation ** exps
    darker = 1.05 ** np.linspace(0, 1, num_bands,)
    absorptions = absorptions * darker
    absorptions = np.minimum(np.maximum(absorptions, 0.0005), 0.99)
    return absorptions

def main(args):
    # open metadata file 
    if not os.path.exists(args.dir_path):
        os.makedirs(args.dir_path)
    filepath = os.path.join(args.dir_path, 'metadata.csv')
    fieldnames = ['filename','sr', 'max_order', 'volume', 'room_dim','source', 'mic', 'rt60']
    
    config = {}
    
    config['sr'] = args.sr

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()   

        for i_rir in tqdm(range(args.n_rir)):
            if i_rir < args.n_rir/2:
                room, config['mic'], config['source'] = get_random_ir(
                    room_size_base=3, 
                    room_size_pow=3,
                    smoothing_range=(3, 5),
                    smoothing_compensate=True,
                    )
            else: 
                room, config['mic'], config['source'] = get_random_ir(
                    room_size_base=5, 
                    room_size_pow=2,
                    smoothing_range=(1, 3),
                    smoothing_compensate=False,
                    )                
            config['room_dim'] = room.shoebox_dim.tolist()
            config['volume'] = room.volume
            config['max_order'] = room.max_order
            config['rt60'] = room.measure_rt60

            config['filename'] = os.path.join(args.dir_path, str(uuid.uuid4()) + '.wav')
            # write rir
            sf.write(config['filename'], room.rir[0][0], room.fs)
            writer.writerow(config)
            csvfile.flush()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_rir', default = 200000, type=int,
        help = 'Number of RIRs to be simulated')
    parser.add_argument('--sr', default = 48000, type=int,
        help='Sampling rate')
    parser.add_argument('--dir_path', 
    help='path to directory where to save the dataset')

    args = parser.parse_args()

    main(args)
