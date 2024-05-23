import argparse
import logging
import os, sys
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import IsolationForest
from FeatureManager import FeatureManager

def setup_logging(log_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)
                        ])

class IsolationForestTrainer:
    def __init__(self, tsv_file, config):
        self.tsv_file = tsv_file
        self.chunk_size = config['chunk_size']
        self.model_path = config['model_path']
        self.feature_manager = FeatureManager()

    def train_model(self):
        try:
            # Try loading the entire dataset
            df = pd.read_csv(self.tsv_file, sep='\t')
            df = self.feature_manager.feature_engineering(df)
            model = IsolationForest()
            model.fit(df)
        except MemoryError:
            logging.info("Dataset too large, processing in chunks...")
            model = IsolationForest()
            for chunk in pd.read_csv(self.tsv_file, sep='\t', chunksize=self.chunk_size):
                chunk = self.feature_manager.feature_engineering(chunk)
                model.fit(chunk)
                self.feature_manager.update_global_state(chunk)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logging.info("Model training completed and saved to %s.", self.model_path)
        print("Model training completed and saved to {}.".format(self.model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Isolation Forest Trainer')
    parser.add_argument('--file', type=str, required=True, help='Path to the TSV file for training the model')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config['log_path'])
    trainer = IsolationForestTrainer(args.file, config)
    trainer.train_model()
