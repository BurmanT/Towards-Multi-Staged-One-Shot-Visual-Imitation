import os
import json 
import copy
import torch
import hydra
import torch.nn as nn
from os.path import join
from omegaconf import OmegaConf
from mosaic.utils.lr_scheduler import build_scheduler
from collections import defaultdict 
torch.autograd.set_detect_anomaly(True)
import learn2learn as l2l
from train_utils import * 
import wandb 

class Trainer:
    def __init__(self, allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device
        def_device = hydra_cfg.device if hydra_cfg.device != -1 else 0
        self._device = torch.device("cuda:{}".format(def_device))
        self._device_list = None
        self._allow_val_grad = allow_val_grad 
        # set of file saving
        assert os.path.exists(self.config.save_path), "Warning! Save path {} doesn't exist".format(self.config.save_path)
        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'

        append = "-Batch{}".format(int(self.config.bsize)) 
        if 'mosaic' in hydra_cfg.policy._target_:
            append = "-Batch{}-{}gpu-Attn{}ly{}-Act{}ly{}mix{}".format(
                int(self.config.bsize), int(torch.cuda.device_count()),
                int(self.config.policy.attn_cfg.n_attn_layers), int(self.config.policy.attn_cfg.attn_ff),
                int(self.config.policy.action_cfg.n_layers), int(self.config.policy.action_cfg.out_dim),
                int(self.config.policy.action_cfg.n_mixtures))

            if self.config.policy.concat_demo_head: 
                append += "-headCat"
            elif self.config.policy.concat_demo_act:
                append += "-actCat"
            else:
                append += "-noCat"
            if 'mosaic' in hydra_cfg.policy._target_:
                append += "-simclr{}x{}".format(int(self.config.policy.simclr_config.compressor_dim), int(self.config.policy.simclr_config.hidden_dim))
            
        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'), str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        self._step = None
        if self.config.wandb_log:
            config_keys = ['train_cfg', 'tasks', 'samplers', 'dataset_cfg', 'policy']
            # for k in config_keys:
            #     print(k, self.config.get(k))
            #     print(k, dict(self.config.get(k)))
            #     print('-'*20)
            wandb_config = {k: self.config.get(k) for k in config_keys}
            run = wandb.init(project='mosaic', name=self.config.exp_name, config=wandb_config)
 
    def train(self, model, weights_fn=None, save_fn=None, optim_weights=None): 
        self._train_loader, self._val_loader = make_data_loaders(self.config, self.train_cfg.dataset)
        # wrap model in DataParallel if needed and transfer to correct device
        print('Training stage \n Found {} GPU devices \n'.format(self.device_count))
        print("Are you training?")
        print(self._device)
        model = model.to(self._device)
        print("script after print")
        if self.device_count > 1 and not isinstance(model, nn.DataParallel):
            print("Training stage \n Device list: {}".format(self.device_list))
            model = nn.DataParallel(model, device_ids=self.device_list)

        # initialize optimizer and lr scheduler
        optim_weights       = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(optim_weights, self.train_cfg)

        # initialize constants:
        epochs              = self.train_cfg.get('epochs', 1)
        vlm_alpha           = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq            = self.train_cfg.get('log_freq', 1000)
        val_freq            = self.train_cfg.get('val_freq', 1000)
        print_freq          = self.train_cfg.get('print_freq', 10000) 
        save_freq           = self.train_cfg.get('save_freq', 10000)

        print("Loss multipliers: \n BC: {} inv: {} Point: {}".format(
            self.train_cfg.bc_loss_mult, self.train_cfg.inv_loss_mult, self.train_cfg.pnt_loss_mult))
        print({name: mul for name, mul in self.train_cfg.rep_loss_muls.items() if mul != 0})
        if self.train_cfg.bc_loss_mult == 0 and self.train_cfg.inv_loss_mult == 0:
            assert sum([v for k, v in self.train_cfg.rep_loss_muls.items()]) != 0, self.train_cfg.rep_loss_muls

        self.tasks          = self.config.tasks
        num_tasks           = len(self.tasks)
        sum_mul             = sum( [task.get('loss_mul', 1) for task in self.tasks] )
        task_loss_muls      = { task.name:
            float("{:3f}".format(task.get('loss_mul', 1) / sum_mul)) for task in self.tasks }
        print(" Weighting each task loss separately:", task_loss_muls)
        self.generated_png  = False
        self._step          = 0
        val_iter            = iter(self._val_loader) 
        # log stats to both 'task_name/loss_name' AND 'loss_name/task_name'
        raw_stats           = dict()
        print(f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}, \
                which sums to {epochs * len(self._train_loader)} total train steps, \
                validation loader has length {len(self._val_loader)}")
        for e in range(epochs):
            frac = e / epochs  
            for inputs in self._train_loader:

                if self._step % save_freq == 0: # save model AND stats
                    self.save_checkpoint(model, optimizer, weights_fn, save_fn)
                    if save_fn is not None:
                        save_fn(self._save_fname, self._step)
                    else:
                        save_module = model
                        if weights_fn is not None:
                            save_module = weights_fn()
                        elif isinstance(model, nn.DataParallel):
                            save_module = model.module
                        torch.save(save_module.state_dict(), self._save_fname + '-{}.pt'.format(self._step))
                    if self.config.get('save_optim', False):
                        torch.save(optimizer.state_dict(), self._save_fname + '-optim-{}.pt'.format(self._step))

                    stats_save_name = join(self.save_dir, 'stats', '{}.json'.format('train_val_stats'))
                    json.dump({k: str(v) for k, v in raw_stats.items()}, open(stats_save_name, 'w'))
                    
                optimizer.zero_grad()
                ## calculate loss here:
                task_losses =  calculate_task_loss(self.config, self.train_cfg, self._device, model, inputs)
                task_names = sorted(task_losses.keys())
                weighted_task_loss = sum([l["loss_sum"] * task_loss_muls.get(name) for name, l in task_losses.items()])
                weighted_task_loss.backward()
                optimizer.step()
                ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
                # calculate train iter stats
                if self._step % log_freq == 0:
                    train_print = collect_stats(self._step, task_losses, raw_stats, prefix='train')   
                    if self.config.wandb_log:
                        tolog = {'Train Step': self._step}
                        for task_name, losses in task_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'train/{loss_name}/{task_name}'] = loss_val
                                tolog[f'train/{task_name}/{loss_name}'] = loss_val
                        wandb.log(tolog)
                    
                    if self._step % print_freq == 0:
                        print('Training epoch {1}/{2}, step {0}: \t '.format(self._step, e, epochs))
                        print(train_print) 

                if self._step % val_freq == 0:
                    # exhaust all data in val loader and take avg loss
                    all_val_losses = {task: defaultdict(list) for task in task_names}
                    val_iter = iter(self._val_loader)
                    model = model.eval()
                    for val_inputs in val_iter:
                        if self.config.use_daml: # allow grad! 
                            val_task_losses = calculate_task_loss(self.confg, self.train_cfg,  self._device, model, val_inputs)
                        else:
                            with torch.no_grad(): 
                                val_task_losses = calculate_task_loss(self.config, self.train_cfg, self._device, model, val_inputs)
       
                        for task, losses in val_task_losses.items():
                            for k, v in losses.items():
                                all_val_losses[task][k].append(v)
                    if self.config.wandb_log:
                        tolog = {'Validation Step': self._step}
                        for task_name, losses in val_task_losses.items():
                            for loss_name, loss_val in losses.items():
                                tolog[f'val/{loss_name}/{task_name}'] = loss_val
                                tolog[f'val/{task_name}/{loss_name}'] = loss_val
                        wandb.log(tolog)
                    # take average across all batches in the val loader 
                    avg_losses = dict()
                    for task, losses in all_val_losses.items():
                        avg_losses[task] = {
                            k: torch.mean(torch.stack(v)) for k, v in losses.items()}
                    
                    val_print = collect_stats(self._step, avg_losses, raw_stats, prefix='val')
                    if self._step % print_freq == 0:
                        print('Validation step {}:'.format(self._step))
                        print(val_print)
                    
                    model = model.train()
                
                self._step += 1
                # update target params
                mod = model.module if isinstance(model, nn.DataParallel) else model
                if self.train_cfg.target_update_freq > -1:
                    mod.momentum_update(frac)
                    if self._step % self.train_cfg.target_update_freq == 0:
                        mod.soft_param_update()
  
        ## when all epochs are done, save model one last time
        self.save_checkpoint(model, optimizer, weights_fn, save_fn)

    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None):
        if save_fn is not None:
            save_fn(self._save_fname, self._step)
        else:
            save_module = model
            if weights_fn is not None:
                save_module = weights_fn()
            elif isinstance(model, nn.DataParallel):
                save_module = model.module
            torch.save(save_module.state_dict(), self._save_fname + '-{}.pt'.format(self._step))
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname + '-optim-{}.pt'.format(self._step))
        print(f'Model checkpoint saved at step {self._step}')
        return 

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    @property
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)
    
    def _build_optimizer_and_scheduler(self, optim_weights, cfg):
        assert self.device_list is not None, str(self.device_list)
        optimizer = torch.optim.Adam(
            optim_weights, cfg.lr, weight_decay=cfg.get('weight_decay', 0))
        return optimizer, build_scheduler(optimizer, cfg.get('lr_schedule', {}))


    def _loss_to_scalar(self, loss):
        """For more readable logging"""
        x = loss.item()
        return float("{:.3f}".format(x))

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0

class Workspace(object):
    """ Initializes the policy model and prepare for Trainer.train() """
    def __init__(self, cfg):
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing trainer")
        config = self.trainer.config
        resume = config.get('resume', False)
        self.action_model = hydra.utils.instantiate(config.policy)
        config.use_daml = 'DAMLNetwork' in cfg.policy._target_
        if config.use_daml:
            print("Switching to l2l.algorithms.MAML")
            self.action_model = l2l.algorithms.MAML(
                self.action_model,
                lr=config['policy']['maml_lr'],
                first_order=config['policy']['first_order'],
                allow_unused=True)

        print("Action model initialized to: {}".format(config.policy._target_))
        if resume:
            rpath = join(cfg.save_path, cfg.resume_path) 
            assert os.path.exists(rpath), "Can't seem to find {} anywhere".format(config.resume_path)
            print('load model from ...%s' % rpath)
            self.action_model.load_state_dict(torch.load(rpath, map_location=torch.device('cpu')))
        
        self.config = config
        self.train_cfg = config.train_cfg

        ## move log path to here!
        print('\n Done initializing Workspace, saving config.yaml to directory: {}'.format(self.trainer.save_dir))

        os.makedirs(self.trainer.save_dir, exist_ok=('burn' in self.trainer.save_dir))
        os.makedirs(join(self.trainer.save_dir, 'stats'), exist_ok=True)
         
        save_config = copy.deepcopy(self.trainer.config)
        OmegaConf.save(config=save_config, f=join(self.trainer.save_dir, 'config.yaml'))


    def run(self):
        self.trainer.train(self.action_model)
        print("Done training")


@hydra.main(
    config_path="experiments", 
    config_name="config.yaml")
def main(cfg): 
    from train_any import Workspace as W
    all_tasks_cfgs = [cfg.tasks_cfgs.nut_assembly, cfg.tasks_cfgs.door, cfg.tasks_cfgs.drawer, cfg.tasks_cfgs.button, cfg.tasks_cfgs.new_pick_place, cfg.tasks_cfgs.stack_block, cfg.tasks_cfgs.basketball]
    
    if cfg.single_task:
        cfg.tasks = [tsk for tsk in all_tasks_cfgs if tsk.name == cfg.single_task]
    
    if cfg.use_all_tasks:
        print("Loading all 7 tasks to the dataset!  obs_T: {} demo_T: {}".format(\
            cfg.dataset_cfg.obs_T, cfg.dataset_cfg.demo_T))
        cfg.tasks = all_tasks_cfgs
    
    if cfg.exclude_task:
        print(f"Training with 6 tasks and exclude {cfg.exclude_task}")
        cfg.tasks = [tsk for tsk in all_tasks_cfgs if tsk.name != cfg.exclude_task]
    
    if cfg.set_same_n > -1:
        for tsk in cfg.tasks:
            tsk.n_per_task = cfg.set_same_n
        cfg.bsize = sum( [tsk.n_tasks * cfg.set_same_n for tsk in cfg.tasks] )
        cfg.vsize = cfg.bsize
        print(f'To construct a training batch, set n_per_task of all tasks to {cfg.set_same_n}, new train/val batch sizes: {cfg.train_cfg.batch_size}/{cfg.train_cfg.val_size}')
        
    if cfg.limit_num_traj > -1:
        print('Only using {} trajectory for each sub-task'.format(cfg.limit_num_traj))
        for tsk in cfg.tasks:
            tsk.traj_per_subtask = cfg.limit_num_traj
    if cfg.limit_num_demo > -1:
        print('Only using {} demon. trajectory for each sub-task'.format(cfg.limit_num_demo))
        for tsk in cfg.tasks:
            tsk.demo_per_subtask = cfg.limit_num_demo 
    
    if 'mosaic' not in cfg.policy._target_:
        print(f'Running baseline method: {cfg.policy._target_}')
        cfg.target_update_freq = -1
    workspace = W(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
    print("Training Done")
