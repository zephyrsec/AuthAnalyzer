import argparse
import logging
import os, sys
import pandas as pd
import pickle
import yaml
from FeatureManager import FeatureManager

def setup_logging(log_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_path),
                            logging.StreamHandler(sys.stdout)
                        ])

class EventAnalyzer:
    def __init__(self, tsv_file, config):
        self.tsv_file = tsv_file
        self.chunk_size = config['chunk_size']
        self.model_path = config['model_path']
        self.output_path = config['anomalous_events_output_path']
        self.feature_manager = FeatureManager()

    def analyze_events(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError("Model file not found. Train the model first.")
            
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Try loading the entire dataset
            df = pd.read_csv(self.tsv_file, sep='\t')
            df = self.feature_manager.feature_engineering(df)
            predictions = model.predict(df)
            anomaly_scores = model.decision_function(df)
            anomalies = df[predictions == -1]
            anomaly_scores = anomaly_scores[predictions == -1]
            
            # Convert anomaly scores to confidence scores
            min_score = min(anomaly_scores)
            max_score = max(anomaly_scores)
            confidence_scores = 100 * (1 - (anomaly_scores - min_score) / (max_score - min_score))

            # Save anomalies with confidence scores to a text file
            with open(self.output_path, 'w') as f:
                for index, row in anomalies.iterrows():
                    f.write(f"{row.to_dict()}, Confidence Score: {confidence_scores[index]}\n")

            logging.info(f"Event analysis completed. Anomalies detected and saved to {self.output_path}")
            print(f"Event analysis completed. Anomalies detected and saved to {self.output_path}")

        except MemoryError:
            logging.info("Dataset too large, processing in chunks...")
            for chunk in pd.read_csv(self.tsv_file, sep='\t', chunksize=self.chunk_size):
                chunk = self.feature_manager.feature_engineering(chunk)
                predictions = model.predict(chunk)
                anomaly_scores = model.decision_function(chunk)
                anomalies = chunk[predictions == -1]
                anomaly_scores = anomaly_scores[predictions == -1]
                
                # Convert anomaly scores to confidence scores
                min_score = min(anomaly_scores)
                max_score = max(anomaly_scores)
                confidence_scores = 100 * (1 - (anomaly_scores - min_score) / (max_score - min_score))

                # Save anomalies with confidence scores to a text file
                with open(self.output_path, 'a') as f:
                    for index, row in anomalies.iterrows():
                        f.write(f"{row.to_dict()}, Confidence Score: {confidence_scores[index]}\n")

            logging.info(f"Event analysis completed. Anomalies detected and saved to {self.output_path}")
            print(f"Event analysis completed. Anomalies detected and saved to {self.output_path}")

        except Exception as e:
            logging.error(f"Error during event analysis: {e}")
            print(f"Error during event analysis: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Event Analyzer')
    parser.add_argument('--file', type=str, required=True, help='Path to the TSV file for analyzing events')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config['log_path'])
    analyzer = EventAnalyzer(args.file, config)
    analyzer.analyze_events()
