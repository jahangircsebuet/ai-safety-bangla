# 3.1 finetuning data preparation
# CUDA_VISIBLE_DEVICES=2 python convert_llamagurad_data_for_finetuning.py

# 3.2 generate undersampled balanced datasets
# CUDA_VISIBLE_DEVICES=2 python generate_undersampled_balanced_dataset.py

# 3.3 generate balanced safety data stats
# CUDA_VISIBLE_DEVICES=2 python safety_data_analysis.py

# 3.4 LoRA finetuning
CUDA_VISIBLE_DEVICES=2 python refusal_lora_finetuner.py