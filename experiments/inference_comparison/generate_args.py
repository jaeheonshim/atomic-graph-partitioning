import sys
import random
from main.partitioner import PARTITIONERS

NUM_ATOMS = (10000, 200000, 10000)
NUM_PARTITIONS = (5, 21, 5)
MESSAGE_PASSING = (1, 6, 1)
NUM_MATERIALS = 3
MODELS = ('mattersim', 'orb')
RESULT_DIR = 'inference/'

def sample_lines(filepath, k):
    reservoir = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if i < k:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = line
    return reservoir

if __name__ == '__main__':
    num_samples = len(MESSAGE_PASSING) * len(MODELS) * len(PARTITIONERS) * ((NUM_ATOMS[1]-NUM_ATOMS[0])//NUM_ATOMS[2]) * ((NUM_PARTITIONS[1]-NUM_PARTITIONS[0])//NUM_PARTITIONS[2]) * NUM_MATERIALS
    materials = sample_lines('mp_ids.txt', NUM_MATERIALS)
    
    print(num_samples, file=sys.stderr)
    
    for n in range(*NUM_ATOMS):
        for num_partitions in range(*NUM_PARTITIONS):
            for material in materials:
                for model in MODELS:
                    for partitioner in PARTITIONERS:
                        for mp in range(*MESSAGE_PASSING):
                            mp_id = material.split(',')[0]
                            print(f'--num_atoms {n} --num_partitions {num_partitions} --mp_id {mp_id} --model {model} --partitioner {partitioner} --message_passing {mp} --result_dir {RESULT_DIR}')