import torch
import pandas as pd
import datetime
import random
import os
   

class ExperimentLogger:
    """Handles logging of experiment results and configurations."""
    def __init__(self, opt):
        self.opt = opt
        self.output_directory = 'chackpoints/{}-{}-{}-{}'.format(self.opt.experiment_name, self.opt.dataset_name, self.opt.attribute_representation_type, self.opt.epochs)
        self.log_file = f'{self.output_directory}/log_{self.opt.experiment_name}.txt'
        self.csv_log_file = f'{self.output_directory}/log_{self.opt.experiment_name}.csv'
        self.history = {
            'epoch': [],
            'accuracy': [],
            'loss': [],
            'best_accuracy': [],
            'best_loss': [],
            'best_accuracy_epoch': [],
            'unseen_accuracy': [],
            'seen_accuracy': [],
            'harmonic_mean': [],
            'best_unseen_accuracy': [],
            'best_seen_accuracy': [],
            'best_harmonic_mean': [],
            'best_harmonic_mean_epoch': [],
            'loss_D': [],
            'loss_G': [],
            'wasserstein_distance': [],
            'real_contrastive_loss': [],
            'fake_contrastive_loss': [],
            'cls_loss_real': [],
            'cls_loss_fake': [],
            'time': []
        }
        os.makedirs(self.output_directory, exist_ok=True)
        self.last_log_text = ""
        
        self.seed()

    def log_print(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def save_csv_history(self):
        pd.DataFrame(self.history).to_csv(self.csv_log_file, index=False)
        self.log_print(f'[INFO] Saved: {self.csv_log_file}')

    def save_model(self, epoch, models, seed, path):
        state = {
            'epoch': epoch + 1,
            'state_dict_generator': models['generator'].state_dict(),
            'state_dict_quality_critic': models['quality_critic'].state_dict(),
            'state_dict_embedding_net': models['embedding_net'].state_dict(),
            'state_dict_discriminator': models['discriminator'].state_dict(),
            'seed': seed,
            'last_log': self.last_log_text,
        }
        torch.save(state, path)
        self.log_print(f'[INFO] Model saved to {path}')

    def resume_training(self, models):
        if self.opt.resume and os.path.isfile(self.opt.resume):
            self.log_print(f"[INFO] Loading checkpoint '{self.opt.resume}'")
            checkpoint = torch.load(self.opt.resume, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            models['generator'].load_state_dict(checkpoint['state_dict_generator'])
            models['quality_critic'].load_state_dict(checkpoint['state_dict_quality_critic'])
            models['embedding_net'].load_state_dict(checkpoint['state_dict_embedding_net'])
            models['discriminator'].load_state_dict(checkpoint['state_dict_discriminator'])
            self.opt.manualSeed = checkpoint['seed']
            start_epoch = checkpoint['epoch']
            self.last_log_text = checkpoint['last_log']
            return start_epoch
        else:
            self.log_print("[WARN]  No checkpoint found at '{}'. Starting from scratch.".format(self.opt.resume))
            return 0
        
    def start_experiment(self, ntrain, device):
        """Print the training settings"""
        self.log_print(f'[INFO]  torch {torch.__version__}  {device}  ({torch.cuda.get_device_name()}, {torch.cuda.get_device_properties(0).total_memory // 1024 ** 2} MiB)')
        self.log_print("[INFO]  training samples: " + str(ntrain))
        self.log_print('[INFO]  Settings: ' + ' '.join(f'{k}={v}' for k, v in vars(self.opt).items()))

    def end_experiment(self, total_time):
        self.log_print('\n\n[INFO] Execution time: ' + str(datetime.timedelta(seconds=total_time)))

    def update_history_and_log(self, metrics):
        # Update history
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)
        # Print and save the log message
        self.log_print(self.format_log(metrics))

    def format_log(self, metrics):
        log_msg = '  '.join(['Epoch: {epoch}/{epochs}', '|',
                    'ZSL: acc: {accuracy:.3f} loss: {loss:.3f} ({best_accuracy:.3f} {best_loss:.3f} @ {best_accuracy_epoch})', '|',
                    'GZSL: U: {unseen:.3f} S: {seen:.3f} H: {harmonic_mean:.3f} ({best_unseen_accuracy:.3f} {best_seen_accuracy:.3f} {best_harmonic_mean:.3f} @ {best_harmonic_mean_epoch})', '|',
                    'Loss: D: {Loss_D:.3f} G: {Loss_G:.3f}', '|',
                    'time: {time}s'])
        
        log_msg = log_msg.format(
            epoch=metrics['epoch'],
            epochs=self.opt.epochs,
            accuracy=metrics['accuracy'],
            loss=metrics['loss'],
            best_accuracy=metrics['best_accuracy'],
            best_loss=metrics['best_loss'],
            best_accuracy_epoch=metrics['best_accuracy_epoch'],
            unseen=metrics['unseen_accuracy'],
            seen=metrics['seen_accuracy'],
            harmonic_mean=metrics['harmonic_mean'],
            best_unseen_accuracy=metrics['best_unseen_accuracy'],
            best_seen_accuracy=metrics['best_seen_accuracy'],
            best_harmonic_mean=metrics['best_harmonic_mean'],
            best_harmonic_mean_epoch=metrics['best_harmonic_mean_epoch'],
            Loss_D=metrics['loss_D'],
            Loss_G=metrics['loss_G'],
            time=metrics['time']
        )
        
        return log_msg
    
    def seed(self):
        """Set up seeding"""
        if self.opt.seed is None:
            self.opt.seed = random.randint(1, 10000)
            self.log_print("[INFO]  random reed: " + str(self.opt.seed))
        else:
            self.log_print("[INFO]  manual reed: " + str(self.opt.seed))

        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed_all(self.opt.seed)
        

class Result:
    """
    Tracks the training progress and results.
    """
    def __init__(self):
        """
        Initializes tracking variables for results.
        """
        self.best_acc = 0.0
        self.best_loss = 0.0
        self.best_epoch = 0
        self.best_acc_s = 0.0
        self.best_acc_u = 0.0
        self.acc_list = []
        self.epoch_list = []
        self.save_model = False

    def update(self, epoch, acc, loss):
        """
        Updates the tracked results for conventional ZSL.

        Args:
            epoch (int): Current epoch number.
            acc (float): Current accuracy.
            loss (float): Current loss.
        """
        self.acc_list.append(acc)
        self.epoch_list.append(epoch)
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_loss = loss
            self.best_epoch = epoch
            self.save_model = True

    def update_gzsl(self, epoch, acc_u, acc_s, h):
        """
        Updates the tracked results for GZSL.

        Args:
            epoch (int): Current epoch number.
            acc_u (float): Accuracy on unseen classes.
            acc_s (float): Accuracy on seen classes.
            H (float): Harmonic mean of acc_u and acc_s.
        """
        self.acc_list.append(h)
        self.epoch_list.append(epoch)
        if h > self.best_acc:
            self.best_acc = h
            self.best_epoch = epoch
            self.best_acc_u = acc_u
            self.best_acc_s = acc_s
            self.save_model = True