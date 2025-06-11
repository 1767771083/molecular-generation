import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import re
from rdkit import RDLogger
import cairosvg
from torch import autograd

from src import sascorer

RDLogger.DisableLog('rdApp.*')
atom_decoder_m = {0: 6, 1: 7, 2: 8, 3: 9}
bond_decoder_m = {1: Chem.rdchem.BondType.SINGLE,
                  2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def flatten_graph_data(adj, x):
    return torch.cat((adj.reshape([adj.shape[0], -1]), x.reshape([x.shape[0], -1])), dim=1)


def split_channel(x):
    n = x.shape[1] // 2
    return x[:, :n], x[:, n:]


def get_graph_data(x, num_nodes, num_relations, num_features):
    """
    Converts a vector of shape [b, num_nodes, m] to Adjacency matrix
    of shape [b, num_relations, num_nodes, num_nodes]
    and a feature matrix of shape [b, num_nodes, num_features].
    :param x:
    :param num_nodes:
    :param num_relations:
    :param num_features:
    :return:
    """
    adj = x[:, :num_nodes*num_nodes *
            num_relations].reshape([-1, num_relations, num_nodes, num_nodes])
    feat_mat = x[:, num_nodes*num_nodes *
                 num_relations:].reshape([-1, num_nodes, num_features])
    return adj, feat_mat


def Tensor2Mol(A, x):
    mol = Chem.RWMol()
    # x[x < 0] = 0.
    # A[A < 0] = -1
    # atoms_exist = np.sum(x, 1) != 0
    atoms = np.argmax(x, 1)
    atoms_exist = atoms != 4
    atoms = atoms[atoms_exist]
    atoms += 6
    adj = np.argmax(A, 0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atom)))

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])

    return mol


def construct_mol(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+] [S+]
            # not support [O-], [N-] [S-]  [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


def construct_mol_with_validation(x, A, atomic_num_list):
    """

    :param x:  (9,5)
    :param A:  (4,9,9)
    :param atomic_num_list: [6,7,8,9,0]
    :return:
    """
    mol = Chem.RWMol()
    # x (ch, num_node)
    atoms = np.argmax(x, axis=1)
    # last a
    atoms_exist = atoms != len(atomic_num_list) - 1
    atoms = atoms[atoms_exist]
    # print('num atoms: {}'.format(sum(atoms>0)))

    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = np.array(adj)
    adj = adj[atoms_exist, :][:, atoms_exist]
    adj[adj == 3] = -1
    adj += 1
    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder_m[adj[start, end]])
            t = adj[start, end]
            while not valid_mol_can_with_seg(mol):
                mol.RemoveBond(int(start), int(end))
                t = t-1
                if t >= 1:
                    mol.AddBond(int(start), int(end), bond_decoder_m[t])

    return mol


def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(
        x, isomericSmiles=True)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s, isomericSmiles=True):
        return s
    return None


def valid_mol_can_with_seg(x, largest_connected_comp=True):
    # mol = None
    if x is None:
        return None
    sm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = Chem.MolFromSmiles(sm)
    if largest_connected_comp and '.' in sm:
        # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm = [(s, len(s)) for s in sm.split('.')]
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    return mol


def check_valency(mol):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
        Chem.SanitizeMol(
            mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(x):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = x
    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append(
                    (b.GetIdx(), int(b.GetBondType()),
                     b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                )
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder_m[t])
                # if '.' in Chem.MolToSmiles(mol, isomericSmiles=True):
                #     print(tt)
                #     print(Chem.MolToSmiles(mol, isomericSmiles=True))

    return mol


def test_correct_mol():
    mol = Chem.RWMol()
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(6))
    mol.AddAtom(Chem.Atom(7))
    mol.AddBond(0, 1, Chem.rdchem.BondType.DOUBLE)
    mol.AddBond(1, 2, Chem.rdchem.BondType.TRIPLE)
    mol.AddBond(0, 3, Chem.rdchem.BondType.TRIPLE)
    print(Chem.MolToSmiles(mol))  # C#C=C#N
    mol = correct_mol(mol)
    print(Chem.MolToSmiles(mol))  # C=C=C=N


def check_tensor(x):
    return valid_mol(Tensor2Mol(*x))


def adj_to_smiles(adj, x, atomic_num_list):
    # adj = _to_numpy_array(adj, gpu)
    # x = _to_numpy_array(x, gpu)
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem, atomic_num_list), isomericSmiles=True)
             for x_elem, adj_elem in zip(x, adj)]
    return valid

def get_data_par(data_name):
    if data_name == 'qm9':
        from data import transform_qm9
        data_file = 'qm9_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_qm9.transform_fn
        atomic_num_list = [6, 7, 8, 9, 0]
        b_n_type = 4
        b_n_squeeze = 3
        a_n_node = 9
        a_n_type = len(atomic_num_list)  # 5
        valid_idx = transform_qm9.get_val_ids()  # len: 13,082, total data: 133,885
    elif data_name == 'zinc250k':
        from data import transform_zinc250k
        data_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list
        b_n_type = 4
        b_n_squeeze = 19  # 2
        a_n_node = 38
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = transform_zinc250k.get_val_ids()
    elif data_name == 'moses':
        from data import transform_moses
        data_file = 'moses_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_moses.transform_fn_moses
        # [6, 7, 8, 9, 16, 17, 35, 0]
        atomic_num_list = transform_moses.moses_atomic_num_list
        b_n_type = 4
        b_n_squeeze = 19  # 2
        a_n_node = 27
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = []  # transform_moses10k.get_val_ids()
    elif data_name == 'bs1':
        from src.data import transform_bs1
        data_file = 'bs1_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_bs1.transform_fn_bs1
        # [5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 35, 53, 0]
        atomic_num_list = transform_bs1.bs1_atomic_num_list
        b_n_type = 4
        a_n_node = 75
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = []
    elif data_name == 'jak2':
        from data import transform_jak2
        data_file = 'jak2_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_jak2.transform_fn_jak2
        # [6, 7, 8, 9, 15, 16, 17, 35, 0]
        atomic_num_list = transform_jak2.jak2_atomic_num_list
        b_n_type = 4
        a_n_node = 53
        a_n_type = len(atomic_num_list)
        valid_idx = []
    elif data_name == 'gen':
        from data import transform_bs1
        data_file = 'gen_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_bs1.transform_fn_bs1
        # [6, 7, 8, 9, 15, 16, 17, 35, 0]
        atomic_num_list = transform_bs1.bs1_atomic_num_list
        b_n_type = 4
        a_n_node = 53
        a_n_type = len(atomic_num_list)
        valid_idx = []
    else:
        raise ValueError(
            'Only support [qm9, zinc250k, moses, bs1, jak2] right now. '
            'Parameters need change a little bit for other dataset.')
    return data_file, transform_fn, atomic_num_list, b_n_type, a_n_node, a_n_type, valid_idx

def get_pred_data_par(data_name):
    from data import transform_prediction
    if data_name == 'bace':
        data_file = 'bace_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_prediction.transform_fn_bace
        atomic_num_list = transform_prediction.bace_atomic_num_list
        b_n_type = transform_prediction.n_bonds
        a_n_node = transform_prediction.bace_max_atoms
        a_n_type = len(atomic_num_list)
        valid_idx = []
    elif data_name == 'delaney':
        data_file = 'delaney_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_prediction.transform_fn_delaney
        atomic_num_list = transform_prediction.delaney_atomic_num_list
        b_n_type = transform_prediction.n_bonds
        a_n_node = transform_prediction.delaney_max_atoms
        a_n_type = len(atomic_num_list)
        valid_idx = []
    elif data_name == 'freesolv':
        data_file = 'freesolv_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_prediction.transform_fn_freesolv
        atomic_num_list = transform_prediction.freesolv_atomic_num_list
        b_n_type = transform_prediction.n_bonds
        a_n_node = transform_prediction.freesolv_max_atoms
        a_n_type = len(atomic_num_list)
        valid_idx = []
    elif data_name == 'clintox':
        data_file = 'clintox_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_prediction.transform_fn_clintox
        atomic_num_list = transform_prediction.clintox_atomic_num_list
        b_n_type = transform_prediction.n_bonds
        a_n_node = transform_prediction.clintox_max_atoms
        a_n_type = len(atomic_num_list)
        valid_idx = []
    elif data_name == 'bbbp':
        data_file = 'bbbp_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_prediction.transform_fn_bbbp
        atomic_num_list = transform_prediction.bbbp_atomic_num_list
        b_n_type = transform_prediction.n_bonds
        a_n_node = transform_prediction.bbbp_max_atoms
        a_n_type = len(atomic_num_list)
        valid_idx = []
    elif data_name == 'lipo':
        data_file = 'lipo_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_prediction.transform_fn_lipo
        atomic_num_list = transform_prediction.lipo_atomic_num_list
        b_n_type = transform_prediction.n_bonds
        a_n_node = transform_prediction.lipo_max_atoms
        a_n_type = len(atomic_num_list)
        valid_idx = []

    else:
        raise ValueError(
            'Only support [bace, delaney, freesolv, clintox, bbbp, lipo] right now. '
            'Parameters need change a little bit for other dataset.')
    return data_file, transform_fn, atomic_num_list, b_n_type, a_n_node, a_n_type, valid_idx


def get_total_label(dataloader, data_name):
    total_real_labels = None
    if data_name == 'qm9':
        total_real_labels = torch.cat([batch[2][:, 0:1] for batch in dataloader], dim=0).float()

    elif data_name == 'zinc250k':
        total_real_labels = torch.cat([batch[2][:, 1:2] for batch in dataloader], dim=0).float()

    elif data_name == 'moses':
        total_real_labels = torch.cat([batch[2] for batch in dataloader], dim=0).float()
    else:
        print('No other datasets are supported !')

    return total_real_labels

def get_real_label(label, data_name, property):
    real_label = None
    if property == 'qed':
        if data_name in ['qm9', 'zinc250k', 'moses']:
            real_label = label[:, 1:2]
        else:
            print('No other datasets are supported !')
    elif property == 'sas':
        if data_name in ['qm9', 'zinc250k', 'moses']:
            real_label = label[:, 2:3]
        else:
            print('No other datasets are supported !')
    elif property == 'pic':
        if data_name in ['bs1', 'jak2']:
            real_label = label[:, 0:1]
        else:
            print('No other datasets are supported !')
    elif property == 'both_train':
        if data_name in ['qm9', 'zinc250k', 'moses']:
            real_label = label[:, 1:3]
        elif data_name in ['bs1', 'jak2']:
            real_label = label[:, [1, 2, 0]]
        else:
            print('No other datasets are supported !')
    else:
        print('Parameter error')

    return real_label


def get_fake_label(label_qed, label_sa, label_pic, size, device):

    fake_label_qed = np.random.uniform(low=label_qed[0], high=label_qed[1], size=size)

    fake_label_sa = np.random.uniform(low=label_sa[0], high=label_sa[1], size=size)

    if label_pic:
        fake_label_pic = np.random.uniform(low=label_pic[0], high=label_pic[1], size=size)
    else:
        fake_label_pic = np.random.uniform(low=label_sa[0], high=label_sa[1], size=size)

    fake_label = np.hstack((fake_label_qed, fake_label_sa, fake_label_pic))

    fake_label = torch.from_numpy(fake_label).float().to(device)

    return fake_label

def simplified_function(x):

    y_values = torch.zeros_like(x)

    y_values[x <= 1] = 1

    mask = (x > 1) & (x < 10)
    y_values[mask] = 1 - torch.log10(x[mask])

    y_values[x >= 10] = 0

    return y_values

def replace_nan_with_row_mean(x):
    """
    检查张量中的NaN值，并将其替换为所在行的平均值。
    参数:
    x: 输入张量，形状为 (batch, dim)
    返回:
    替换后的张量
    """
    x_clone = x.clone()
    # 找到NaN的位置
    nan_mask = torch.isnan(x_clone)
    # 计算每行的平均值，忽略NaN
    row_mean = torch.nanmean(x_clone, dim=1, keepdim=True)
    # 将NaN替换为对应行的平均值
    x_clone[nan_mask] = row_mean.expand_as(x_clone)[nan_mask]

    return x_clone

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    参数:
    - discriminator: discriminator 网络
    - real_samples: 真实样本
    - fake_samples: 生成样本
    返回:
    - gradient_penalty: 梯度罚项
    """
    # 生成一个随机插值系数 epsilon
    alpha = torch.rand(real_samples.size(0), 1, 1, device=real_samples.device)
    alpha = alpha.expand_as(real_samples)

    # 在真实样本和生成样本之间生成插值样本
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)

    # 设置 interpolates 为需要计算梯度
    interpolates.requires_grad_(True)

    # 计算 critic 对插值样本的输出
    d_interpolates = discriminator(interpolates,
                                   torch.randint(low=0,
                                                 high=discriminator.total_time_step,
                                                 size=(real_samples.shape[0], 1),
                                                 device=real_samples.device),
                                   apply_noise=False
                                   )

    # 生成与 d_interpolates 形状相同的全 1 的张量
    fake = torch.ones_like(d_interpolates, device=real_samples.device)

    # 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # 计算梯度的 L2 范数
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def check_validity(adj, x, atomic_num_list, gpu=-1, return_unique=True,
                   correct_validity=True, largest_connected_comp=True, debug=True, print_information=True):
    # True
    """

    :param adj:  (100,4,9,9)
    :param x: (100.9,5)
    :param atomic_num_list: [6,7,8,9,0]
    :param gpu:  e.g. gpu0
    :param return_unique:
    :return:
    """
    adj = _to_numpy_array(adj)  # , gpu)  (1000,4,9,9)
    x = _to_numpy_array(x)  # , gpu)  (1000,9,5)
    if correct_validity:
        # valid = [valid_mol_can_with_seg(construct_mol_with_validation(x_elem, adj_elem, atomic_num_list)) # valid_mol_can_with_seg
        #          for x_elem, adj_elem in zip(x, adj)]
        valid = []
        for x_elem, adj_elem in zip(x, adj):
            mol = construct_mol(x_elem, adj_elem, atomic_num_list)
            # Chem.Kekulize(mol, clearAromaticFlags=True)
            cmol = correct_mol(mol)
            # valid_mol_can_with_seg(cmol)  # valid_mol(cmol)  # valid_mol_can_with_seg
            vcmol = valid_mol_can_with_seg(
                cmol, largest_connected_comp=largest_connected_comp)
            # Chem.Kekulize(vcmol, clearAromaticFlags=True)
            valid.append(vcmol)
    else:
        valid = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
                 for x_elem, adj_elem in zip(x, adj)]  # len()=1000
    # len()=valid number, say 794
    valid = [mol for mol in valid if mol is not None]
    if print_information:
        print("valid molecules: {}/{}".format(len(valid), adj.shape[0]))
    n_mols = x.shape[0]
    valid_ratio = len(valid)/n_mols  # say 794/1000
    valid_smiles = [Chem.MolToSmiles(
        mol, isomericSmiles=False) for mol in valid]
    unique_smiles = list(set(valid_smiles))  # unique valid, say 788
    unique_ratio = 0.
    if len(valid) > 0:
        unique_ratio = len(unique_smiles)/len(valid)  # say 788/794
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    abs_unique_ratio = len(unique_smiles)/n_mols
    if print_information:
        print("valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}%".
              format(valid_ratio * 100, unique_ratio * 100, abs_unique_ratio * 100))
    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles
    results['valid_ratio'] = valid_ratio*100
    results['unique_ratio'] = unique_ratio*100
    results['abs_unique_ratio'] = abs_unique_ratio * 100

    return results


def check_novelty(gen_smiles, train_smiles, n_generated_mols):  # gen: say 788, train: 120803
    if len(gen_smiles) == 0:
        novel_ratio = 0.
        abs_novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        novel_ratio = novel*100./len(gen_smiles)  # 743*100/788=94.289
        abs_novel_ratio = novel*100./n_generated_mols
    print("novelty: {:.3f}%, abs novelty: {:.3f}%".format(
        novel_ratio, abs_novel_ratio))
    return novel_ratio, abs_novel_ratio

def calculate_sa_score(mol):
    try:
        if mol is None:
            raise ValueError("Invalid SMILES string")
        sa_score = sascorer.calculateScore(mol)
        return sa_score
    except Exception as e:
        print(f"Error calculating SA score for SMILES: {e}")
        return None

def calculate_reward(qed_score, logp_score, vality):
    """计算奖励函数"""
    if not vality:
        vality_penalty = -0.1  # 给予一个负向奖励，表示不合法分子的惩罚
    else:
        vality_penalty = 0.0  # 分子有效时不给予额外的惩罚

    # 奖励函数 = QED 分数 - LogP 分数
    reward = qed_score - logp_score + vality_penalty
    return reward



def _to_numpy_array(a):  # , gpu=-1):
    if isinstance(a, torch.Tensor):
        a = a.cpu().detach().numpy()
    # if gpu >= 0:
    #     return cuda.to_cpu(a)
    elif isinstance(a, np.ndarray):
        # We do not use cuda np.ndarray in pytorch
        pass
    else:
        raise TypeError("a ({}) is not a torch.Tensor".format(type(a)))
    return a

def get_loss_curve(discriminator_losses, generator_losses, predictor_losses, data_name, result_dir):

    loss_subdir = 'loss'
    loss_dir = os.path.join(result_dir, loss_subdir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    # GAN Losses
    plt.figure(figsize=(10, 6))
    plt.plot(discriminator_losses, label='Discriminator Loss')
    plt.plot(generator_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Losses')

    # Save the plot
    plt.savefig(os.path.join(loss_dir, f"{data_name}_GAN_losses.png"))

    # GAE and Predictor Losses
    plt.figure(figsize=(10, 6))
    # plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.plot(predictor_losses, label='Predictor Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Predictor Losses')

    # Save the plot
    plt.savefig(os.path.join(loss_dir, f"{data_name}_Predictor_losses.png"))


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)

def plot_top_mol(df, top_name):   # create the image of molecule
    vmol = []
    vlabel = []
    for index, row in df.head(n=100).iterrows():
        score, smile, sim, smile_old = row
        vmol.append(Chem.MolFromSmiles(smile))
        vlabel.append('{:.3f}'.format(score))

    svg = Draw.MolsToGridImage(vmol, legends=vlabel, molsPerRow=10,  # 5,
                               subImgSize=(240, 240), useSVG=True)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to=f"top_{top_name}.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to=f"top_{top_name}.png")

    # svg = Draw.MolsToGridImage(
    #     top_qed_mols,
    #     legends=top_qed_values,
    #     molsPerRow=10,  # 5,
    #     subImgSize=(240, 240),
    #     useSVG=True,
    #     maxFontSize=24)
    #
    # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
    #                  write_to=os.path.join(args.save_dir, "top_qed.pdf"))
    # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
    #                  write_to=os.path.join(args.save_dir, "top_qed.png"))
    # svg = Draw.MolsToGridImage(
    #     top_logp_mols,
    #     legends=top_logp_values,
    #     molsPerRow=10,  # 5,
    #     subImgSize=(240, 240),
    #     useSVG=True)  # , useSVG=True
    #
    # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
    #                  write_to=os.path.join(args.save_dir, "top_logp.pdf"))
    # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
    #                  write_to=os.path.join(args.save_dir, "top_logp.png"))


if __name__ == '__main__':

    test_correct_mol()
