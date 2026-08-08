
import torch
import numpy as np
import os
import pickle

from sde_minimax_model import SDEMinimaxModel


def get_patient_data(data_source, patient_id):
    try:
        seq_len = int(data_source['sequence_lengths'][patient_id])
        max_len = data_source['cancer_volume'].shape[1]
        if seq_len > max_len:
            print(f"Warning: Patient {patient_id} has sequence length {seq_len}, exceeding the data dimension; truncating to {max_len}.")
            seq_len = max_len
        
        patient_data = {
            'id': patient_id,
            'cancer_volume': data_source['cancer_volume'][patient_id, :seq_len],
            'chemo_application': data_source['chemo_application'][patient_id, :seq_len],
            'radio_application': data_source['radio_application'][patient_id, :seq_len],
            'patient_type': int(data_source['patient_types'][patient_id])
        }
        return patient_data
    except IndexError:
        print(f"Error: Index out of range while extracting patient {patient_id}; skipping this patient.")
        return None

def filter_golden_cohort(training_data, ids_to_check):
    print("\n[Substep] Selecting the golden cohort for core training...")
    golden_ids = []
    
    for i in ids_to_check:
        try:
            seq_len = int(training_data['sequence_lengths'][i])
            
            if seq_len <= 30:
                continue
                
            chemo_history = training_data['chemo_application'][i, :seq_len]
            radio_history = training_data['radio_application'][i, :seq_len]
            if np.sum(chemo_history) == 0 or np.sum(radio_history) == 0:
                continue
                
            volume_history = training_data['cancer_volume'][i, :seq_len]
            min_vol = np.min(volume_history)
            max_vol = np.max(volume_history)
            
            if min_vol < 1e-6:
                if max_vol < 1.0:
                    continue
            elif max_vol / min_vol < 5.0:
                continue
                
            golden_ids.append(i)
        except IndexError:
            continue
            
    print(f"Selected {len(golden_ids)} patients for the golden cohort.")
    return golden_ids


def train_model_for_dataset(data_path, output_model_path):
    print(f"\n{'='*80}")
    print(f"Processing dataset: {os.path.basename(data_path)}")
    print(f"{'='*80}")

    print("\n[Step 1/4] Loading data...")
    try:
        with open(data_path, 'rb') as f:
            training_data = pickle.load(f)
        
        print(f"Loaded {len(training_data['patient_types'])} patient records.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}; skipping it.")
        return
    except Exception as e:
        print(f"Error: Failed to load or parse {data_path}: {e}. Check the file contents and structure.")
        return

    print("\n[Step 2/4] Initializing the SDE model (v5.0 DR-AIPW)...")
    params = {
        'noise_dim': 10, 'hidden_dim': 64, 'lr': 1e-4, 
        'lambda_fpe': 1.0, 'lambda_ic': 10.0, 'lambda_obs': 500.0,
        'lambda_treatment': 100.0, 'epochs': 300, 'batch_size': 256
    }
    sde_model = SDEMinimaxModel(fpe_solver_params=params)
    print("Model initialized.")

    print("\n[Step 3/4] Grouping patients by type...")
    grouped_patient_ids = {1: [], 2: [], 3: []}
    for i in range(len(training_data['patient_types'])):
        p_type = int(training_data['patient_types'][i])
        if p_type in grouped_patient_ids:
            grouped_patient_ids[p_type].append(i)
    
    print(f"Patient counts: type 1: {len(grouped_patient_ids[1])}, type 2: {len(grouped_patient_ids[2])}, type 3: {len(grouped_patient_ids[3])}")

    final_model_state_for_dataset = {}

    for p_type in sorted(grouped_patient_ids.keys()):
        print(f"\n--- Training patient type {p_type} ---")
        
        all_ids_for_type = grouped_patient_ids[p_type]
        if not all_ids_for_type:
            print(f"No patients of type {p_type} were found; skipping.")
            continue
        
        patient_ids_to_train = filter_golden_cohort(training_data, all_ids_for_type)
        
        if not patient_ids_to_train:
            print("Warning: The golden cohort is empty; using all patients of this type.")
            patient_ids_to_train = all_ids_for_type

        print(f"Final training cohort for type {p_type}: {len(patient_ids_to_train)} patients.")

        all_training_patients_for_type = []
        for patient_id in patient_ids_to_train:
            patient_data = get_patient_data(training_data, patient_id)
            if patient_data and len(patient_data['cancer_volume']) > 1:
                all_training_patients_for_type.append(patient_data)
        
        if all_training_patients_for_type:
            print(f"Training type {p_type} on {len(all_training_patients_for_type)} patients...")
            sde_model.train(all_training_patients_for_type)
            print(f"Training completed for type {p_type}.")
            
            final_model_state_for_dataset[p_type] = sde_model.Drift_Nets[p_type].state_dict()
        else:
            print(f"No valid training data remained for type {p_type}; skipping.")

    if final_model_state_for_dataset:
        print(f"\nSaving the combined model to: {output_model_path}")
        torch.save(final_model_state_for_dataset, output_model_path)
        print("Model saved successfully.")
    else:
        print("\nNo model was trained successfully for this dataset; no model file was created.")


def main():
    data_dir = os.environ.get('CT_CQL_DATA_DIR', os.path.join(os.path.dirname(__file__), 'data'))
    output_dir = os.environ.get('CT_CQL_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), 'outputs'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    dataset_to_model_map = {
        'can_sim_chemo_treatment_train.p': 'chemo_treatment_model.pth',
        'can_sim_no_treatment_train.p': 'no_treatment_model.pth',
        'can_sim_radio_chemo_train.p': 'radio_chemo_model.pth',
        'can_sim_radio_treatment_train.p': 'radio_treatment_model.pth'
    }

    for data_filename, model_filename in dataset_to_model_map.items():
        data_path = os.path.join(data_dir, data_filename)
        output_path = os.path.join(output_dir, model_filename)
        
        train_model_for_dataset(data_path=data_path, output_model_path=output_path)
        
    print(f"\n{'*'*80}")
    print("All training tasks are complete.")
    print(f"Generated model files are available in: {output_dir}")
    print(f"{'*'*80}")


if __name__ == '__main__':
    main()
