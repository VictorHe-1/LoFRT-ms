import os

import mindspore
import torch


def pytorch_params(pth_file, verbose=False):
    par_dict = torch.load(pth_file, map_location='cpu')
    pt_params = {}
    if 'model' in par_dict and len(par_dict) < 10:
        par_dict = par_dict['model']
    elif 'state_dict' in par_dict:
        par_dict = par_dict['state_dict']

    for name, value in par_dict.items():
        if verbose:
            print(name, value.numpy().shape)
        pt_params[name] = value.numpy()
    return pt_params


def mindspore_params(network, verbose=False):
    ms_params = {}
    for param in network.get_parameters():
        name = param.name
        value = param.data.asnumpy()
        if verbose:
            print(name, value.shape)
        ms_params[name] = value
    return ms_params


def mapper(ms_name: str):
    m2t = dict(
        gamma='weight',
        beta='bias',
        moving_mean='running_mean',
        moving_variance='running_var'
    )
    for k in m2t.keys():
        if k not in ms_name:
            continue
        return ms_name.replace(k, m2t[k])

    return ms_name


def map_torch_to_mindspore(ms_dict, torch_dict, verbose=False):
    new_params_list = []
    for name, value in ms_dict.items():
        torch_name = mapper(name)
        torch_value = torch_dict[torch_name]

        convert_value = mindspore.Tensor(torch_value)

        if verbose:
            print(name, value.shape)
        assert value.shape == convert_value.shape, f"value shape not match, ms {name} {value.shape}, torch {torch_name}{convert_value.shape}"
        new_params_list.append(dict(name=name, data=convert_value))
    return new_params_list


def convert_parameter(i_pth_path, i_ms_pth_path, ms_model, verbose=False):
    print('\n' * 2, '**-------------below are torch------------**', '\n' * 2)
    pt_param = pytorch_params(i_pth_path, verbose=verbose)
    print('\n' * 2, '**-----------below are mindspore----------**', '\n' * 2)
    ms_param = mindspore_params(ms_model, verbose=verbose)
    print('\n' * 2, '**---------below are after convert--------**', '\n' * 2)
    ms_params_list = map_torch_to_mindspore(ms_param, pt_param, verbose=verbose)

    print(f'\nsuccessfully convert the checkpoint, saved as {i_ms_pth_path}')
    mindspore.save_checkpoint(ms_params_list, i_ms_pth_path)
    mindspore.load_checkpoint(i_ms_pth_path, ms_model)
    print(f'successfully load checkpoint into network')

    # compare
    print(f'\nunit test')
    print(f'torch loftr_fine.layers.0.norm1.bias', pt_param[f'loftr_fine.layers.0.norm1.bias'][:5])

    print(f'ms loftr_fine.layers.0.norm1.beta', ms_model.loftr_fine.layers[0].norm1.beta[:5])


if __name__ == "__main__":
    print("plese input your pytorch loftr model path(e.g. /data1/outdoor_ds.ckpt):", end='')
    pth_path = input()
    ms_pth_path = os.path.join('./models', "ms_outdoor_ds.ckpt")

    from src.models import LoFTR, default_cfg

    model = LoFTR(config=default_cfg)

    convert_parameter(pth_path, ms_pth_path, ms_model=model, verbose=True)
