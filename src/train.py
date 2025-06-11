import subprocess

def run_training_on_all_datasets():
    datasets = ['qm9', 'zinc250k', 'moses']
    for data in datasets:
        print(f"\n--- Training on dataset: {data} ---")
        subprocess.run([
            'python', 'main.py',
            '--data_name', data,
        ])

if __name__ == '__main__':
    run_training_on_all_datasets()







