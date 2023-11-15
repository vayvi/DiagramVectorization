


python scripts/generate_contour_images.py --input_folder images_resized --text_mask_folder images_resized_testr_mask --output_folder images_resized_contours
python scripts/generate_predictions.py --config_path config/config_baseline.yaml --save_plots
python scripts/generate_predictions.py --config_path scripts/config_baseline_no_testr.yaml --save_plots
