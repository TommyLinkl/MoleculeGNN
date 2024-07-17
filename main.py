import os, sys, time, subprocess, itertools, copy, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import json
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from src.models_gnn import SchNet, MEGNet
from src.qm9_process import QM9Dataset
from src.train_eval import train, evaluate, plot_training_validation_cost

def construct_GNN_model(config, data_instance):
    default_SchNet_params = {
        'dim1': 64,
        'dim2': 64,
        'dim3': 64,
        'cutoff': 8,
        'pre_fc_count': 1,
        'gc_count': 3,
        'post_fc_count': 1,
        'pool': "global_mean_pool",
        'dropout_rate': 0.0,
    }
    default_MEGNet_params = {
        'dim1': 64,
        'dim2': 64,
        'dim3': 64,
        'pre_fc_count': 1,
        'gc_count': 3,
        'gc_fc_count': 2, 
        'post_fc_count': 1,
        'pool': "global_mean_pool",
        'dropout_rate': 0.0,
    }

    if config['Model']['model'] == 'SchNet': 
        model_params = {key: config['Model'][key] for key in default_SchNet_params.keys() if key in config['Model']}
        model = SchNet(data_instance, **model_params)
    elif config['Model']['model'] == 'MEGNet': 
        model_params = {key: config['Model'][key] for key in default_MEGNet_params.keys() if key in config['Model']}
        model = MEGNet(data_instance, **model_params)
    else: 
        raise NotImplementedError(f"Model {config['Model']['model']} not implemented.")

    if (config['Job']['load_model']=='True'):
        checkpoint = torch.load(f"{config['Job']['output_directory']}init_model.pth")

        # Remove 'module.' prefix from keys if model was trained with DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)        # model.load_state_dict(checkpoint['state_dict'])
        print("The GNN model is initialized by loading from the existing state_dict file.")
    
    return model 


def run_training_mode(config, device):
    # Load the QM9 dataset
    dataset = QM9Dataset(root='./data/QM9', batch_size=config['Model']['batch_size'])
    dataset.process_dataset(target=config['Job']['target_index'])
    # dataset.print_instance()
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    torch.cuda.empty_cache()

    # Construct the GNN model
    model = construct_GNN_model(config, next(iter(train_loader)))
    if (device.type == 'cuda') or (device.type == 'mps'):
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer, scheduler, and loss definition
    optimizer = getattr(optim, config['Model']['optimizer'])(model.parameters(), lr=config['Model']['lr'], **config['Model']["optimizer_args"])
    scheduler = getattr(optim.lr_scheduler, config['Model']['scheduler'])(optimizer, **config['Model']['scheduler_args'])
    criterion = nn.MSELoss()

    # Training loop
    training_COST = []
    validation_COST = []

    best_val_error = float('inf')
    best_epoch = 0

    file_trainCost = open(f"{config['Job']['output_directory']}training_cost.dat", 'w')
    file_valCost = open(f"{config['Job']['output_directory']}validation_cost.dat", 'w')

    for epoch in range(config['Model']['num_epochs']):
        start_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        end_time = time.time()
        scheduler.step(val_loss)

        training_COST.append(train_loss)
        validation_COST.append(val_loss)
        file_trainCost.write(f"{epoch + 1}  {train_loss}\n")
        file_trainCost.flush()
        file_valCost.write(f"{epoch + 1}  {val_loss}\n")
        file_valCost.flush()
        print(f"Epoch {epoch + 1}/{config['Model']['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}. Runtime: {end_time - start_time:.3f}s")
        torch.cuda.empty_cache()
        
        # Save model checkpoint every 100 epochs
        if epoch%100==0:
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'full_model': model,
            }, f"{config['Job']['output_directory']}epoch_{epoch + 1}.pth")

        # Save model checkpoint if validation loss improves
        if val_loss < best_val_error:
            best_val_error = val_loss
            best_epoch = epoch
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'full_model': model,
            }, f"{config['Job']['output_directory']}best_model.pth")

        # Early stopping
        if epoch - best_epoch > config['Model']['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch + 1} as validation loss has not improved for {config['Model']['early_stopping_patience']} epochs.")
            break
    file_trainCost.close()
    file_valCost.close()

    # Plotting training and validation costs
    fig_cost = plot_training_validation_cost(training_COST, validation_COST, False)
    fig_cost.savefig(f"{config['Job']['output_directory']}train_val_cost.pdf")
    fig_cost = plot_training_validation_cost(training_COST, validation_COST, True)
    fig_cost.savefig(f"{config['Job']['output_directory']}train_val_cost_log.pdf")
    torch.cuda.empty_cache()

    # Load best model for evaluation
    checkpoint = torch.load(f"{config['Job']['output_directory']}best_model.pth")
    model.load_state_dict(checkpoint['state_dict'])
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss of the best set of parameters: {test_loss:.4f}")
    torch.cuda.empty_cache()

    return test_loss


def run_predict_mode():
    return


def train_func_for_tune(one_config_simplified, device=None):
    # one_config_simplified only contains the ['Model'] section
    temp_one_config = {
        'Job': {'load_model': "False"}, 
        'Model': one_config_simplified, 
    }

    # Load the QM9 dataset
    dataset = QM9Dataset(root='./data/QM9', batch_size=temp_one_config['Model']['batch_size'])
    dataset.process_dataset(target=temp_one_config['Job']['target_index'])
    # dataset.print_instance()
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    torch.cuda.empty_cache()

    # Construct the GNN model
    model = construct_GNN_model(temp_one_config, next(iter(train_loader)))
    if device.type in ['cuda', 'mps']:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer, scheduler, and loss definition
    optimizer = getattr(optim, one_config_simplified['optimizer'])(model.parameters(), lr=one_config_simplified['lr'], **one_config_simplified["optimizer_args"])
    scheduler = getattr(optim.lr_scheduler, one_config_simplified['scheduler'])(optimizer, **one_config_simplified['scheduler_args'])
    criterion = nn.MSELoss()
    torch.cuda.empty_cache()

    # Training loop
    training_COST = []
    validation_COST = []
    best_val_error = float('inf')
    best_epoch = 0

    for epoch in range(one_config_simplified['num_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        training_COST.append(train_loss)
        validation_COST.append(val_loss)
        torch.cuda.empty_cache()

        # Update scheduler
        tune.report(loss=val_loss)
        
        # Early stopping
        if val_loss < best_val_error:
            best_val_error = val_loss
            best_epoch = epoch

        if epoch - best_epoch > one_config_simplified['early_stopping_patience']:
            print(f"Early stopping at epoch {epoch + 1} as validation loss has not improved for {one_config_simplified['early_stopping_patience']} epochs.")
            break


def run_hyperparameter_mode(config, device): 
    # construct tune_config from config that is read in, extract the "Model" part of the config
    model_config = config['Model']
    tune_config = {}

    for key, value in model_config.items():
        if isinstance(value, list):
            tune_config[key] = tune.choice(value)
        else:
            tune_config[key] = value
    # print(tune_config)

    ray.init()

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=500,
        grace_period=30,
        reduction_factor=2
    )

    # Start hyperparameter tuning with Ray Tune
    analysis = tune.run(
        tune.with_parameters(train_func_for_tune, device=device),
        config=tune_config,
        num_samples=config['Job']['hyper_trials'],
        scheduler=scheduler,
        resources_per_trial={"gpu": 1},  # Use 1 GPU per trial
        checkpoint_at_end=True,  # Enable default checkpoint saving
        verbose=1, 
        resume=(config['Job']['hyper_resume']=='True'),   # Resume from a previous run if True
        local_dir=config['Job']['output_directory'], 
        max_concurrent_trials=1
    )

    # Get best hyperparameters
    best_config = analysis.get_best_config(metric="loss", mode="min")

    # Print best hyperparameters
    print("Best hyperparameters found:")
    print(best_config)

    # Shutdown Ray
    ray.shutdown()

    return


def run_hyperparameter_mode_manual(config): 
    # Check if any value in config['Job'] is a list
    for key, value in config['Job'].items():
        if isinstance(value, list):
            raise ValueError(f"List found in config['Job'][{key}]. Only config['Model'] can contain lists.")

    # Extract model configurations and generate combinations
    model_keys = config['Model'].keys()
    model_values = [config['Model'][key] if isinstance(config['Model'][key], list) else [config['Model'][key]] for key in model_keys]
    combinations = list(itertools.product(*model_values))
    print(f"We are creating a total of {len(combinations)} hyperparameter trials. ")

    # Create configuration files for each trial
    for i, combo in enumerate(combinations):
        trial_dir = f"{config['Job']['output_directory']}trial_{i}/"
        os.makedirs(trial_dir, exist_ok=True)
        one_config_path = f"{trial_dir}config.yml"
    
        # Change Job settings for each one_config
        one_config = copy.deepcopy(config)
        one_config['Job']['run_mode'] = "Training"
        one_config['Job']['output_directory'] = trial_dir
        one_config['Job']['clear_dir'] = 0
        if 'hyper_trials' in one_config['Job']:
            del one_config['Job']['hyper_trials']
        if 'hyper_resume' in one_config['Job']:
            del one_config['Job']['hyper_resume']
        one_config['Model'] = dict(zip(model_keys, combo))

        with open(one_config_path, 'w') as f:
            yaml.dump(one_config, f)

        # Create a submit_tune_job.sh script
        with open('submit_job_manualTune.sh', 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -A m2651_g
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 6:00:00
#SBATCH -J GNN_manual_tune_trial_{i}
#SBATCH -o {trial_dir}GNN_manual_tune_trial_{i}.log
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
##SBATCH --mail-user=tommy_lin@berkeley.edu
##SBATCH --mail-type=ALL

export SLURM_CPU_BIND="cores"
srun python main.py {one_config_path}
""")

        # Submit jobs using sbatch
        subprocess.run(['sbatch', 'submit_job_manualTune.sh'])
        # time.sleep(3)
    os.remove('submit_job_manualTune.sh')

    return


def run_inference_mode(config, device, writeFileAs='JSON'):
    dataset = QM9Dataset(root='./data/QM9', batch_size=1)
    dataset.process_dataset(target=config['Job']['target_index'], procSMILES=True)
    loader, _, _ = dataset.get_dataloaders(split_ratio=(1.0, 0.0, 0.0))
    torch.cuda.empty_cache()
    # dataset.print_instance()

    if config['Job']['target_index'] == 0:
        property = 'dipMom'
    elif config['Job']['target_index'] == 4: 
        property = 'gap'

    model = construct_GNN_model(config, next(iter(loader)))
    if device.type in ['cuda', 'mps']:
        model = nn.DataParallel(model)
    model = model.to(device)

    start_time = time.time()
    model.eval()
    results = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)

            # Process each data instance in the batch
            result = {
                'idx': data.idx[0].item(), 
                'name': data.name[0], 
                'atom_types': data.z.cpu().tolist(), 
                'num_atoms': data.num_nodes, 
                f'ref_{property}': data.y[0,0].item(), 
                f'pred_{property}': output[0].item(),
            }
            results.append(result)
    end_time = time.time()
    print(f"Inference on all molecules. Runtime: {end_time - start_time:.3f}s")

    start_time = time.time()
    inference_results_df = pd.DataFrame(results)
    inference_results_df = inference_results_df.sort_values(by='idx')
    if writeFileAs=='JSON':
        # Save the DataFrame to a JSON file
        inference_results_json = inference_results_df.to_json(orient='records')

        # Custom JSON encoder to handle the specific formatting
        class CustomJSONEncoder(json.JSONEncoder):
            def iterencode(self, obj, _one_shot=False):
                if isinstance(obj, list):
                    return '[' + ','.join(json.dumps(el) for el in obj) + ']'
                else:
                    return super().iterencode(obj, _one_shot)

        # Use the custom JSON encoder to format the output
        formatted_json = json.dumps(json.loads(inference_results_json), indent=4, cls=CustomJSONEncoder)

        # Save the formatted JSON string to a file
        with open(f"{config['Job']['output_directory']}inference_results.json", 'w') as f:
            f.write(formatted_json)
    elif writeFileAs=='CSV':
        # Save DataFrame to CSV
        inference_results_df.to_csv(f"{config['Job']['output_directory']}inference_results.csv", index=False)
    else: 
        raise NotImplementedError("Only support writing to JSON or CSV files. ")
    end_time = time.time()
    print(f"Writing inference results to file. Runtime: {end_time - start_time:.3f}s")

    return inference_results_df



def main(config_filename):
    # NVIDIA, CUDA support / MPS (Apple Silicon GPU) support
    if torch.cuda.is_available():
        print("CUDA GPU is available. ")
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s). \n")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS is available. ")
        device = torch.device('mps')
        num_gpus = 1  # MPS typically uses a single GPU
        print("Using 1 GPU (MPS). \n")
    else:
        print("Only CPU is available. This will be slow. \n")
        device = torch.device('cpu')
    
    # Reading calculation information + hyperparameters
    with open(config_filename, 'r') as file:
        config = yaml.safe_load(file)
    print(f"Done reading in the {config_filename} file. The job type is {config['Job']['run_mode']}. The model is {config['Model']['model']}.")
    if bool(config['Job']['clear_dir']):
        try:
            if os.path.exists(f"{config['Job']['output_directory']}"):
                shutil.rmtree(f"{config['Job']['output_directory']}")
                print(f"Directory {config['Job']['output_directory']} and its contents successfully deleted\n")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error occurred while deleting directory {config['Job']['output_directory']}: {e}\n")
    os.makedirs(config['Job']['output_directory'], exist_ok = True)
    
    if config['Job']['run_mode'] == 'Training':
        test_loss = run_training_mode(config, device)
    elif config['Job']['run_mode'] == 'Hyperparameter':
        # run_hyperparameter_mode(config, device)
        run_hyperparameter_mode_manual(config)
    elif config['Job']['run_mode'] == 'Inference':
        run_inference_mode(config, device)
    else:
        raise NotImplementedError(f"The job run_mode of '{config['Job']['run_mode']}' is not implemented.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_filename>")
        sys.exit(1)

    config_filename = sys.argv[1]
    main(config_filename)
