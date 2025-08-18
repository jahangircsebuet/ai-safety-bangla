# 3.1 load dataset from Anthropic dataset in HF - 16000 samples
# CUDA_VISIBLE_DEVICES=2 python load_data_from_anthropic.py

# just for testing llamaguard classifier, do not use for finetuning 
# CUDA_VISIBLE_DEVICES=2 python llamaguard_as_classifier.py


# 3.2 this is used for safe/unsafe classification using LlamaGuard 7B Model - 16000 samples
# CUDA_VISIBLE_DEVICES=2 python classify_data_using_llamaguard_7b.py

# 3.3 translate the llamagurad english data into bangla - 16000 samples
# CUDA_VISIBLE_DEVICES=2 python translate_llamaguard_dataset_into_bangla.py

# CUDA_VISIBLE_DEVICES=2 python test_env.py