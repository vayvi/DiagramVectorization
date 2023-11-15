

python helper/parse_svg.py --data_path data/diagrams 
python helper/preprocess_data.py data/diagrams data/diagrams_processed
python evaluation/generate_gt.py --data_path data/diagrams

