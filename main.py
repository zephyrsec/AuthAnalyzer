import argparse
import logging
import os
import sys
from IsolationForestTrainer import IsolationForestTrainer
from EventAnalyzer import EventAnalyzer

def setup_logging():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_directory, 'event_analyzer.log')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler(sys.stdout)
                        ])

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description='Authentication Event Analyzer')
    parser.add_argument('--trainmodel', type=str, help='Path to the TSV file for training the model')
    parser.add_argument('--analyze', type=str, help='Path to the TSV file for analyzing events')
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Models')

    if args.trainmodel:
        logging.info("Training model...")
        print("Training model...")
        os.makedirs(models_dir, exist_ok=True)
        trainer = IsolationForestTrainer(args.trainmodel)
        trainer.train_model()
    elif args.analyze:
        if not os.path.exists(models_dir):
            logging.error("Models folder not found. Train the model first.")
            print("Models folder not found. Train the model first.")
            sys.exit(1)
        logging.info("Analyzing events...")
        print("Analyzing events...")
        analyzer = EventAnalyzer(args.analyze)
        analyzer.analyze_events()
    else:
        logging.error("Invalid argument provided. Use --trainmodel or --analyze with a TSV file path.")
        print("Invalid argument provided. Use --trainmodel or --analyze with a TSV file path.")
        sys.exit(1)

if __name__ == '__main__':
    main()
