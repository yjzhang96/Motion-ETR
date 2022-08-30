from email.policy import strict
import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(gpu_ids)>1:
            torch.save(network.cpu().module.state_dict(), save_path)
        else:  
            torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if len(self.gpu_ids)>1:
            network.module.load_state_dict(torch.load(save_path))
        else:
            self.load_GPUS(network,save_path)
        print('---------load model %s'%save_path)
        
    def load_GPUS(self,model,model_path):
        state_dict = torch.load(model_path)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "module" in k:
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)
        return model

    def update_learning_rate():
        pass
