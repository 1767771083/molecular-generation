import os
import argparse

def get_parent_dir():
    return os.path.dirname(os.path.dirname(__file__))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='zinc250k')
    # qm9, zinc250k, moses, bace, delaney, freesolv, clintox
    parser.add_argument('--fine_data_name', type=str, default='bs1')  # bs1, jak2
    parser.add_argument('--data_dir', type=str, default='E:\Desktop\AED-GAN\data', help='Choose dataset dir')
    # '../data'  'E:\Desktop\AED-GAN\data\pred_data'
    parser.add_argument('--save_dir', type=str, default='out', help='Choose save dir')
    parser.add_argument('--result_dir', type=str, default=os.path.join(get_parent_dir(), 'result'))
    parser.add_argument('--eval_subdir', type=str, default='evaluation_metrics', help='Choose eval dir')
    parser.add_argument('--test_subdir', type=str, default='result/test', help='Choose test dir')
    parser.add_argument('--model_dir', type=str, default='../logs', help='Model parameter saving path')

    parser.add_argument('--pretrain_epochs', default=100, help='Train epoch of fine tuning')
    parser.add_argument('--max_epochs', default=300, help='Train epoch')
    parser.add_argument('--batch_size', default=128, help='Batch size')
    # Feature dimensions
    parser.add_argument('--latent_size', default=32, help='Feature dimensions after graph encoding')
    parser.add_argument('--hidden_size', default=128, help='The hidden dimension of the graph encoder')
    # GAN
    parser.add_argument('--label_qed', type=float, nargs=2, default=[0.1, 0.99], help='QED label value range')
    parser.add_argument('--label_sa', type=float, nargs=2, default=[1.0, 9.9], help='SA label value range')
    parser.add_argument('--n_critic', default=1, help='Discriminator update frequency')
    parser.add_argument('--total_time_step', default=1000, help='Total number of steps in the forward diffusion chain')
    parser.add_argument('--lambda_gp', default=10.0, help='Gradient_penalty')
    parser.add_argument('--attr_weight_1', default=1.6, help='QED and SA Property Regularization Penalty')
    parser.add_argument('--attr_weight_2', default=0.5, help='Other Property Regularization Penalty')
    parser.add_argument('--weight_pro', default=0.8, help='The weights between different attributes')
    parser.add_argument('--gamma', default=1.0, help='sigmoid r')
    parser.add_argument('--threshold', default=5.0, help='sigmoid mid')

    # Predictor
    parser.add_argument('--num_encoder', default=2, help='Number of attention encoder layers')
    parser.add_argument('--num_heads', default=8, help='The number of heads in multi-head attention')
    parser.add_argument('--num_ffn', default=128, help='Number of hidden layers in MLP')
    parser.add_argument('--num_neurons', default=256, help='Number of hidden layers in MLP')
    parser.add_argument('--weight_property', default=1.6, help='The weight between two optimization properties')

    parser.add_argument('--pro_pred', default=[2, 8, 64, 128], help='proPredictor')

    args = parser.parse_args()

    args.save_dir = args.save_dir + '_' + args.data_name
    args.save_dir = os.path.join(args.result_dir, args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    return args
