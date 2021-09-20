import os
import torch


class Saver:

    def __init__(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.model_path = model_path
        self.model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        self.model_folder = os.path.join(os.path.dirname(self.model_path), self.model_name)
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

    def save_checkpoint(self, model, optimizer, epoch):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(state, self.model_path)
        e_model_path = os.path.join(self.model_folder, "{}_e{}.ckpt".format(self.model_name, epoch))
        torch.save(state, e_model_path)
        os.chmod(self.model_path, 0o774)
        os.chmod(e_model_path, 0o774)
        print("=> saving checkpoint '{}'".format(e_model_path))


def load_checkpoint_pretrained(model_path, model):
    # Check working device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if not os.path.isfile(model_path):
        print("=> no checkpoint found at '{}'".format(model_path))
        return model

    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    for k, v in checkpoint['state_dict'].items():
        cpt = {}
        cpt[k] = v
        try:
            model_dict.update(cpt)
            model.load_state_dict(model_dict)
        except RuntimeError:
            pass
    print("=> loaded checkpoint '{}'".format(model_path))

    return model


def _get_mapping_key(state_dict):
    mapping_dict = {}
    for key in state_dict:
        mapping_dict[key] = ".".join(key.split('.')[1:])
    return mapping_dict

def _map_keys(state_dict):
    mapping_dict = _get_mapping_key(state_dict)
    all_keys = list(state_dict.keys())
    for key in all_keys:
        try:
            new_key = mapping_dict[key]
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
        except KeyError:
            continue
    return state_dict

def load_checkpoint(model_path, model, optimizer=None, epoch=None):
    # Check working device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if not os.path.isfile(model_path):
        print("=> no checkpoint found at '{}'".format(model_path))
        return

    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        model.load_state_dict(_map_keys(checkpoint['state_dict']))

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, epoch))
    return model, optimizer, epoch
