import os
import argparse

def get_parent_dir():
    return os.path.dirname(os.path.dirname(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='qm9')
    # qm9, zinc250k, moses
    parser.add_argument('--data_dir', type=str, default='data', help='Choose dataset dir')
    parser.add_argument('--save_dir', type=str, default='out', help='Choose save dir')
    parser.add_argument('--result_dir', type=str, default=os.path.join(get_parent_dir(), 'results'))
    parser.add_argument('--model_dir', type=str, default=os.path.join(get_parent_dir(), 'log'))

    parser.add_argument('--max_epochs', default=200, help='Train epoch')
    parser.add_argument('--batch_size', default=64, help='Batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    # Feature dimensions
    parser.add_argument('--node_dim', default=5)
    parser.add_argument('--hidden_dim', default=64)
    parser.add_argument('--time_emb_dim', default=128)

    parser.add_argument('--total_time_step', default=1000, help='Total number of steps in the forward diffusion chain')
    parser.add_argument('--attr_weight_1', default=1.6, help='Property Regularization Penalty')





    args = parser.parse_args()

    args.save_dir = args.save_dir + '_' + args.data_name
    args.save_dir = os.path.join(args.result_dir, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    return args
