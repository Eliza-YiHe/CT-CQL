
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, noise_dim, cov_dim, treatment_dim, hidden_dim, patient_type_embedding_dim=8):
        super(Generator, self).__init__()
        self.patient_type_embed = nn.Embedding(4, patient_type_embedding_dim)
        input_dim = 1 + noise_dim + cov_dim + treatment_dim + patient_type_embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, t, z, covs, treatments, patient_types):
        patient_type_embedded = self.patient_type_embed(patient_types.long())
        full_input = torch.cat([t, z, covs, treatments, patient_type_embedded], dim=1)
        return self.net(full_input)

class Discriminator(nn.Module):
    def __init__(self, cov_dim, treatment_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + 1 + cov_dim + treatment_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, t, y, covs, treatments):
        return self.net(torch.cat([t, y, covs, treatments], dim=1))

class EnhancedDriftNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnhancedDriftNet, self).__init__()
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU()
        )
        self.treatment_net = nn.Sequential(
            nn.Linear(4, 16), nn.ELU(),
            nn.Linear(16, 16), nn.ELU(),
            nn.Linear(16, 8), nn.ELU()
        )
        self.combine_net = nn.Sequential(
            nn.Linear(hidden_dim + 8, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.l2_alpha = 0.01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x, treatments):
        return self.combine_net(torch.cat([self.base_net(x), self.treatment_net(treatments)], dim=1))
    
    def l2_regularization(self):
        return self.l2_alpha * sum(torch.norm(param, p=2)**2 for param in self.parameters())

class PropensityNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(PropensityNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 4), nn.Softmax(dim=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x): 
        return self.net(x)

def train_or_load_propensity_net(all_patient_data, force_retrain=False):
    filepath = 'propensity_net.pth'
    if not force_retrain and os.path.exists(filepath):
        logger.info(f"Loading the existing propensity model from: {filepath}")
        model = PropensityNet().to(DEVICE)
        model.load_state_dict(torch.load(filepath, map_location=DEVICE))
        return model

    logger.info("Training a new propensity model on all patient data...")
    all_covariates, all_treatment_indices = [], []
    for patient in all_patient_data:
        y_real, chemo, radio = patient['cancer_volume'], patient['chemo_application'], patient['radio_application']
        cum_chemo, cum_radio = np.cumsum(chemo), np.cumsum(radio)
        for t in range(1, len(y_real)):
            y_cov, cum_treat_cov = y_real[t-1], cum_chemo[t-1] + cum_radio[t-1]
            all_covariates.append([y_cov, patient['patient_type'], cum_treat_cov])
            chemo_t, radio_t = chemo[t], radio[t]
            if chemo_t == 0 and radio_t == 0: idx = 0
            elif chemo_t == 1 and radio_t == 0: idx = 1
            elif chemo_t == 0 and radio_t == 1: idx = 2
            else: idx = 3
            all_treatment_indices.append(idx)
    
    cov_scaler = MinMaxScaler()
    cov_norm = cov_scaler.fit_transform(np.array(all_covariates))
    treatments_np = np.array(all_treatment_indices)
    
    model = PropensityNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    epochs, batch_size = 2000, 256
    
    best_loss = float('inf')
    for epoch in tqdm(range(epochs), desc="Training PropensityNet", leave=False):
        model.train()
        idx = np.random.permutation(len(cov_norm))
        X_shuffled, Y_shuffled = cov_norm[idx], treatments_np[idx]
        epoch_loss = 0
        for i in range(0, len(X_shuffled), batch_size):
            X_batch = torch.tensor(X_shuffled[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            Y_batch = torch.tensor(Y_shuffled[i:i+batch_size], dtype=torch.long).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= (len(X_shuffled) / batch_size)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), filepath)
        
        if (epoch + 1) % 100 == 0: 
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    logger.info(f"Propensity model saved to {filepath}; best loss: {best_loss:.6f}")
    model.load_state_dict(torch.load(filepath))
    return model


class SDEMinimaxModel:
    def __init__(self, fpe_solver_params=None):
        default_params = {
            'noise_dim': 10, 
            'hidden_dim': 128, 
            'lr': 1e-4, 
            'lambda_fpe': 1.0,
            'lambda_ic': 10.0, 
            'lambda_obs': 1.0,
            'epochs': 100, 
            'batch_size': 256,
            'sde_sigma': 0.01,
            'dr_clip_value': 10.0
        }
        
        if fpe_solver_params is None:
            self.params = default_params
        else:
            self.params = default_params.copy()
            self.params.update(fpe_solver_params)
        
        self.sde_sigma = self.params['sde_sigma']
        
        self.G = Generator(
            noise_dim=self.params['noise_dim'], 
            cov_dim=3, 
            treatment_dim=4,
            hidden_dim=self.params['hidden_dim']
        ).to(DEVICE)
        self.D = Discriminator(
            cov_dim=3, 
            treatment_dim=4, 
            hidden_dim=self.params['hidden_dim']
        ).to(DEVICE)
        
        self.Drift_Nets = { 
            ptype: EnhancedDriftNet(input_dim=4, hidden_dim=self.params['hidden_dim']).to(DEVICE) 
            for ptype in [1, 2, 3] 
        }
        
        lr = self.params['lr']
        self.optimizer_G = optim.Adam(
            self.G.parameters(), 
            lr=lr, 
            betas=(0.5, 0.9),
            weight_decay=1e-5
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(), 
            lr=lr, 
            betas=(0.5, 0.9),
            weight_decay=1e-5
        )
        self.optimizer_Drifts = { 
            ptype: optim.Adam(
                net.parameters(), 
                lr=lr, 
                betas=(0.5, 0.9),
                weight_decay=1e-5
            ) 
            for ptype, net in self.Drift_Nets.items() 
        }
        
        self.propensity_net = None
        self.scalers = {}
        self.epoch_losses = []

    def save_model(self, file_path='sde_model_generalizable.pth'):
        logger.info(f"\nSaving generalizable models to {file_path}...")
        torch.save({
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'drift_nets': {ptype: net.state_dict() for ptype, net in self.Drift_Nets.items()},
            'scalers': self.scalers
        }, file_path)
        logger.info("Generalizable models and scalers saved successfully.")

    def load_model(self, file_path='sde_model_generalizable.pth'):
        if not os.path.exists(file_path):
            logger.warning(f"Warning: Model file not found at {file_path}. Models are not loaded.")
            return False
        logger.info(f"Loading generalizable models from {file_path}...")
        state = torch.load(file_path, map_location=DEVICE)
        self.G.load_state_dict(state['generator'])
        self.D.load_state_dict(state['discriminator'])
        for ptype, state_dict in state['drift_nets'].items():
            self.Drift_Nets[int(ptype)].load_state_dict(state_dict)
        self.scalers = state['scalers']
        logger.info("Generalizable models and scalers loaded successfully.")
        return True

    def _prepare_training_data(self, all_patient_data):
        logger.info("Flattening all patient data for batch training...")
        flat_data, all_y_for_scaler, all_covs_for_scaler = [], [], []

        for patient in all_patient_data:
            y, chemo, radio = patient['cancer_volume'], patient['chemo_application'], patient['radio_application']
            cum_chemo, cum_radio = np.cumsum(chemo), np.cumsum(radio)
            
            log_y = np.log1p(y)
            
            for t in range(1, len(y)):
                y_cov = log_y[t-1]
                cum_treat_cov = cum_chemo[t-1] + cum_radio[t-1]
                covs_raw = [y_cov, patient['patient_type'], cum_treat_cov]
                
                chemo_t, radio_t = chemo[t], radio[t]
                if chemo_t == 0 and radio_t == 0: treat_idx = 0
                elif chemo_t == 1 and radio_t == 0: treat_idx = 1
                elif chemo_t == 0 and radio_t == 1: treat_idx = 2
                else: treat_idx = 3

                flat_data.append({
                    't': float(t),
                    'y_t-1': log_y[t-1],
                    'y_t': log_y[t],
                    'treatment_idx': treat_idx,
                    'patient_type': patient['patient_type'],
                    'covs_raw': covs_raw,
                    'y_0': log_y[0]
                })
                all_y_for_scaler.append(log_y[t])
                all_covs_for_scaler.append(covs_raw)
        
        all_y_for_scaler = np.array(all_y_for_scaler).reshape(-1, 1)
        all_covs_for_scaler = np.array(all_covs_for_scaler)
        
        y_scaler = MinMaxScaler().fit(all_y_for_scaler)
        cov_scaler = MinMaxScaler().fit(all_covs_for_scaler)
        self.scalers = {'y': y_scaler, 'covs': cov_scaler}
        
        logger.info(f"Data preprocessing completed with {len(flat_data)} samples")
        return flat_data

    def _compute_dr_loss(self, y_obs, y_gen, covs_norm, treatments):
        with torch.no_grad():
            prop_scores = self.propensity_net(covs_norm)
        
        prob_A = torch.sum(prop_scores * treatments, dim=1).clamp(min=1e-6)
        
        small_prob_mask = prob_A < 0.01
        if torch.any(small_prob_mask):
            small_prob_count = torch.sum(small_prob_mask).item()
        
        # has_small_prob = torch.any(prob_A < 0.01)
        
        ipw = (y_obs.squeeze() - y_gen.squeeze()) / prob_A
        
        clip_value = 10.0
        if hasattr(self.params, 'dr_clip_value'):
            clip_value = self.params['dr_clip_value']
        elif 'dr_clip_value' in self.params:
            clip_value = self.params['dr_clip_value']
        
        ipw = torch.clamp(ipw, -clip_value, clip_value)
        
        y_hat_aipw = y_gen.squeeze() + ipw
        
        loss_dr = torch.mean((y_hat_aipw - y_obs.squeeze())**2)
        
        return loss_dr

    def train(self, all_patient_data, force_retrain_propensity=False):
        self.propensity_net = train_or_load_propensity_net(all_patient_data, force_retrain_propensity)
        self.propensity_net.eval()
        
        flat_data = self._prepare_training_data(all_patient_data)
        
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']
        total_batches = (len(flat_data) // batch_size + (1 if len(flat_data) % batch_size != 0 else 0))
        
        for epoch in range(epochs):
            np.random.shuffle(flat_data)
            epoch_loss_G, epoch_loss_D = 0.0, 0.0
            epoch_loss_fpe, epoch_loss_ic, epoch_loss_dr = 0.0, 0.0, 0.0
            
            pbar = tqdm(range(0, len(flat_data), batch_size), desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for i in pbar:
                batch = flat_data[i:i+batch_size]
                if not batch: continue

                t_b = torch.tensor([s['t'] for s in batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                t_b.requires_grad = True
                
                y_prev_raw_b = torch.tensor([s['y_t-1'] for s in batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                y_curr_raw_b = torch.tensor([s['y_t'] for s in batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                y_0_raw_b = torch.tensor([s['y_0'] for s in batch], dtype=torch.float32).unsqueeze(1).to(DEVICE)
                
                covs_raw_b = torch.tensor([s['covs_raw'] for s in batch], dtype=torch.float32).to(DEVICE)
                patient_type_b = torch.tensor([s['patient_type'] for s in batch], dtype=torch.int64).to(DEVICE)
                treat_idx_b = torch.tensor([s['treatment_idx'] for s in batch], dtype=torch.int64).to(DEVICE)
                treatments_b = nn.functional.one_hot(treat_idx_b, num_classes=4).float().to(DEVICE)

                y_obs_b = torch.tensor(
                    self.scalers['y'].transform(y_curr_raw_b.cpu().numpy()), 
                    dtype=torch.float32
                ).to(DEVICE)
                covs_norm_b = torch.tensor(
                    self.scalers['covs'].transform(covs_raw_b.cpu().numpy()), 
                    dtype=torch.float32
                ).to(DEVICE)
                
                self.optimizer_D.zero_grad()
                z = torch.randn(len(batch), self.params['noise_dim']).to(DEVICE)
                with torch.no_grad():
                    y_gen_norm_b = self.G(t_b, z, covs_norm_b, treatments_b, patient_type_b)
                
                y_gen_norm_b = y_gen_norm_b.detach().requires_grad_(True)
                
                mu_list = []
                for j in range(len(batch)):
                    ptype = patient_type_b[j].item()
                    drift_input = torch.cat([y_gen_norm_b[j].unsqueeze(0), covs_norm_b[j].unsqueeze(0)], dim=1)
                    mu_list.append(self.Drift_Nets[ptype](drift_input, treatments_b[j].unsqueeze(0)))
                mu_b = torch.cat(mu_list)

                d_output = self.D(t_b, y_gen_norm_b, covs_norm_b, treatments_b)
                
                grad_d_t, grad_d_y_norm = torch.autograd.grad(
                    outputs=d_output, 
                    inputs=[t_b, y_gen_norm_b], 
                    grad_outputs=torch.ones_like(d_output), 
                    create_graph=True,
                    retain_graph=True
                )
                
                grad_d_yy_norm = torch.autograd.grad(
                    outputs=grad_d_y_norm, 
                    inputs=y_gen_norm_b, 
                    grad_outputs=torch.ones_like(grad_d_y_norm), 
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                y_scale = self.scalers['y'].scale_[0]
                fpe_residual = -grad_d_t - mu_b * (grad_d_y_norm / y_scale) - 0.5 * self.sde_sigma**2 * (grad_d_yy_norm / (y_scale**2))
                loss_D = -torch.mean(fpe_residual**2)
                
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(self.D.parameters(), 5.0)
                self.optimizer_D.step()
                for p in self.D.parameters():
                    p.data.clamp_(-0.01, 0.01)
                epoch_loss_D += loss_D.item()

                self.optimizer_G.zero_grad()
                for opt in self.optimizer_Drifts.values(): opt.zero_grad()

                z = torch.randn(len(batch), self.params['noise_dim']).to(DEVICE)
                y_gen_norm_b = self.G(t_b, z, covs_norm_b, treatments_b, patient_type_b)
                
                mu_list = []
                l2_reg = torch.tensor(0.0).to(DEVICE)
                for j in range(len(batch)):
                    ptype = patient_type_b[j].item()
                    drift_input = torch.cat([y_gen_norm_b[j].unsqueeze(0), covs_norm_b[j].unsqueeze(0)], dim=1)
                    mu_j = self.Drift_Nets[ptype](drift_input, treatments_b[j].unsqueeze(0))
                    mu_list.append(mu_j)
                    l2_reg += self.Drift_Nets[ptype].l2_regularization()
                mu_b = torch.cat(mu_list)
                l2_reg /= len(batch)

                d_output = self.D(t_b, y_gen_norm_b, covs_norm_b, treatments_b)
                
                grad_d_t, grad_d_y_norm = torch.autograd.grad(
                    outputs=d_output, 
                    inputs=[t_b, y_gen_norm_b], 
                    grad_outputs=torch.ones_like(d_output), 
                    create_graph=True,
                    retain_graph=True
                )
                
                grad_d_yy_norm = torch.autograd.grad(
                    outputs=grad_d_y_norm, 
                    inputs=y_gen_norm_b, 
                    grad_outputs=torch.ones_like(grad_d_y_norm), 
                    create_graph=True,
                    retain_graph=True
                )[0]
                
                y_scale = self.scalers['y'].scale_[0]
                fpe_residual = -grad_d_t - mu_b * (grad_d_y_norm / y_scale) - 0.5 * self.sde_sigma**2 * (grad_d_yy_norm / (y_scale**2))
                loss_G_fpe = torch.mean(fpe_residual**2)
                epoch_loss_fpe += loss_G_fpe.item()

                y_0_norm_b = torch.tensor(
                    self.scalers['y'].transform(y_0_raw_b.cpu().numpy()), 
                    dtype=torch.float32
                ).to(DEVICE)
                
                z0 = torch.randn(len(batch), self.params['noise_dim']).to(DEVICE)
                t0 = torch.zeros_like(t_b)
                y_gen_t0_norm = self.G(t0, z0, covs_norm_b, treatments_b, patient_type_b)
                loss_G_ic = torch.mean((y_gen_t0_norm - y_0_norm_b)**2)
                epoch_loss_ic += loss_G_ic.item()

                loss_dr_obs = self._compute_dr_loss(y_obs_b, y_gen_norm_b, covs_norm_b, treatments_b)
                epoch_loss_dr += loss_dr_obs.item()
                
                loss_G = (
                    self.params['lambda_fpe'] * loss_G_fpe + 
                    self.params['lambda_ic'] * loss_G_ic + 
                    self.params['lambda_obs'] * loss_dr_obs + 
                    l2_reg
                )
                
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), 2.0)
                for net in self.Drift_Nets.values():
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                
                self.optimizer_G.step()
                for opt in self.optimizer_Drifts.values(): opt.step()
                
                epoch_loss_G += loss_G.item()
                pbar.set_postfix({
                    "Loss_G": f"{loss_G.item():.4f}", 
                    "Loss_D": f"{loss_D.item():.4f}",
                    "FPE": f"{loss_G_fpe.item():.4f}",
                    "IC": f"{loss_G_ic.item():.4f}",
                    "DR": f"{loss_dr_obs.item():.4f}"
                })
            
            epoch_loss_G /= total_batches
            epoch_loss_D /= total_batches
            epoch_loss_fpe /= total_batches
            epoch_loss_ic /= total_batches
            epoch_loss_dr /= total_batches
            self.epoch_losses.append((epoch_loss_G, epoch_loss_D, epoch_loss_fpe, epoch_loss_ic, epoch_loss_dr))
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Loss_G={epoch_loss_G:.4f}, "
                f"Loss_D={epoch_loss_D:.4f}, "
                f"FPE={epoch_loss_fpe:.4f}, "
                f"IC={epoch_loss_ic:.4f}, "
                f"DR={epoch_loss_dr:.4f}"
            )
            
            if (epoch + 1) % 10 == 0:
                self.save_model()
                
                if epoch_loss_dr > 1000.0 and self.params['lambda_obs'] > 0.1:
                    new_lambda = self.params['lambda_obs'] * 0.8
                    logger.warning(f"DR loss is high ({epoch_loss_dr:.2f}); reducing lambda_obs from {self.params['lambda_obs']} to {new_lambda}")
                    self.params['lambda_obs'] = new_lambda

    def predict_distribution(self, patient_data, n_samples=1000):
        self.G.eval()
        for net in self.Drift_Nets.values(): net.eval()

        if not self.scalers:
            raise RuntimeError("Scalers are not fitted. Please train the model first or load them.")

        with torch.no_grad():
            T = len(patient_data['cancer_volume'])
            y_log = np.log1p(patient_data['cancer_volume'])
            
            simulated_y_log = np.zeros((n_samples, T))
            simulated_y_log[:, 0] = y_log[0]

            patient_type = patient_data['patient_type']
            chemo = patient_data['chemo_application']
            radio = patient_data['radio_application']
            cum_chemo = np.cumsum(chemo)
            cum_radio = np.cumsum(radio)
            
            treatment_indices = []
            for t in range(T):
                if t == 0:
                    treatment_indices.append(0)
                else:
                    chemo_t, radio_t = chemo[t], radio[t]
                    if chemo_t == 0 and radio_t == 0: idx = 0
                    elif chemo_t == 1 and radio_t == 0: idx = 1
                    elif chemo_t == 0 and radio_t == 1: idx = 2
                    else: idx = 3
                    treatment_indices.append(idx)
            
            for t in range(1, T):
                y_prev_log_t = simulated_y_log[:, t-1]
                covs_raw = np.zeros((n_samples, 3))
                
                for i in range(n_samples):
                    cum_treat = cum_chemo[t-1] + cum_radio[t-1] if t > 0 else 0
                    covs_raw[i] = [y_prev_log_t[i], patient_type, cum_treat]
                
                covs_norm = torch.tensor(
                    self.scalers['covs'].transform(covs_raw), 
                    dtype=torch.float32
                ).to(DEVICE)
                
                treatments_t = nn.functional.one_hot(
                    torch.tensor([treatment_indices[t]] * n_samples), 
                    num_classes=4
                ).float().to(DEVICE)
                
                t_tensor = torch.full((n_samples, 1), float(t)).to(DEVICE)
                z = torch.randn(n_samples, self.params['noise_dim']).to(DEVICE)
                ptype_tensor = torch.full((n_samples,), patient_type, dtype=torch.int64).to(DEVICE)
                
                y_next_norm = self.G(t_tensor, z, covs_norm, treatments_t, ptype_tensor)
                y_next_log = self.scalers['y'].inverse_transform(y_next_norm.cpu().numpy())
                simulated_y_log[:, t] = y_next_log.squeeze()
        
        final_y_pred = np.expm1(simulated_y_log)
        final_y_pred[final_y_pred < 0] = 0
        
        self.G.train()
        for net in self.Drift_Nets.values(): net.train()
        
        return final_y_pred
