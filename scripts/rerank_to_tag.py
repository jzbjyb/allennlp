import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, root_dir)
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert rerank model weight to tag model weight')
    parser.add_argument('--inp', type=str, help='model file', required=True)
    parser.add_argument('--out', type=str, help='output file, where extractions should be written.', required=True)
    args = parser.parse_args()

    w = torch.load(args.inp, map_location='cpu')
    map_key = {
        'tag_projection_layer_mt._module.weight': 'tag_projection_layer._module.weight',
        'tag_projection_layer_mt._module.bias': 'tag_projection_layer._module.bias',
    }
    rm_key = ['_task', 'score_layer.weight', 'score_layer.bias']
    for ok, nk in map_key.items():
        w[nk] = w[ok]
        del w[ok]
    for ok in rm_key:
        del w[ok]
    torch.save(w, args.out)