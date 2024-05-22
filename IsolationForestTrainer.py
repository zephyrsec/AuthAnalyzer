import argparse
import logging
import os
import sys
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest

def setup_logging():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_directory, 'event_analyzer.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])

class IsolationForestTrainer:
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file

    def train_model(self):
        try:
            df = pd.read_csv(self.tsv_file, sep='\t')
            model = IsolationForest()
            model.fit(df)
            
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'isolation_forest_model.pkl')

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logging.info("Model training completed and saved to %s.", model_path)
            print("Model training completed and saved to {}.".format(model_path))
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            print(f"Error during model training: {e}")

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='Isolation Forest Trainer')
    parser.add_argument('--file', type=str, required=True, help='Path to the TSV file for training the model')
    args = parser.parse_args()

    trainer = IsolationForestTrainer(args.file)
    trainer.train_model()
