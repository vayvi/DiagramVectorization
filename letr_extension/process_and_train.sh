
epochs1=400

python helper/preprocess_data.py data/synthetic_raw data/synthetic_processed
python main.py --config_path config_primitives.yaml --epochs $epochs1

epochs2=200
python main.py --config_path config_primitives_stage2.yaml --epochs $epochs2