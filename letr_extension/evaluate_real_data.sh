

epochs1=400

python main.py --config_path config_primitives.yaml --epochs $epochs1 --test --coco_path data/diagrams_processed
python evaluation/evaluate.py --real_data --epoch $epochs1 --exp_folder res50_stage1_circles

epochs2=200

python main.py --config_path config_primitives_stage2.yaml --epochs $epochs2 --test --coco_path data/diagrams_processed
python evaluation/evaluate.py --real_data --epoch $epochs2 --exp_folder res50_stage2_circles
