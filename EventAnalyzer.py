import argparse
import logging
import os
import sys
import pandas as pd
import pickle

def setup_logging():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_directory, 'event_analyzer.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])

class EventAnalyzer:
    def __init__(self, tsv_file):
        self.tsv_file = tsv_file

    def analyze_events(self):
        try:
            df = pd.read_csv(self.tsv_file, sep='\t')
            
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')
            model_path = os.path.join(models_dir, 'isolation_forest_model.pkl')

            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found. Train the model first.")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            predictions = model.predict(df)
            anomalies = df[predictions == -1]
            
            logging.info("Event analysis completed. Anomalies detected:")
            logging.info(anomalies)
            print("Event analysis completed. Anomalies detected:")
            print(anomalies)
        except Exception as e:
            logging.error(f"Error during event analysis: {e}")
            print(f"Error during event analysis: {e}")

if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='Event Analyzer')
    parser.add_argument('--file', type=str, required=True, help='Path to the TSV file for analyzing events')
    args = parser.parse_args()

    analyzer = EventAnalyzer(args.file)
    analyzer.analyze_events()
