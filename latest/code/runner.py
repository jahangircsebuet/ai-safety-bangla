import sys
import json
import os
import pandas as pd
from pathlib import Path
import json
import re
import unicodedata
from pathlib import Path
# setproctitle("runner.py")
from difflib import SequenceMatcher
from setproctitle import setproctitle
from typing import Any, Dict, List, Tuple
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from response_generator import (
    generate_llama, 
    generate_qwen,
    generate_responses,
    generate_gemma_responses, 
    generate_qwen_responses,
    generate_mistral_mixtral_faster
)
from translate import translate


if __name__ == "__main__":

    dataset_base_path = "/home/malam10/projects/ai-safety-bangla/latest/dataset"

    # Ground Truth dataset filepaths
    toxic_dataset_path = f"{dataset_base_path}/toxic.json"
    hatespeech_dataset_path = f"{dataset_base_path}/hatespeech.json"
    multijail_dataset_path = f"{dataset_base_path}/multijail.json"
    catqa_dataset_path = f"{dataset_base_path}/catqa.json"
    aegis_dataset_path = f"{dataset_base_path}/aegis.json"

    ##########################################################
    ################## Dataset Analysis starts ###############
    ##########################################################
    from dataset_analysis import (
        load_many, 
        flatten_agg, 
        dedup_per_source, 
        flatten_per_source,
        analysis_completeness, 
        analysis_selected_model_distribution,
        run_all_analyses
        )
    
    
    # usage of analysis functions 
    # Example: load all your ground-truth formatted json files
    objects = load_many([toxic_dataset_path, hatespeech_dataset_path, multijail_dataset_path, catqa_dataset_path, aegis_dataset_path])

    if False:
        
        # [DONE] Prepare Aggregated Full Dataset
        from dataset_analysis import flatten_agg, dedup_per_source, flatten_per_source
        agg_output_filepath = f"{dataset_base_path}/df_agg.json"
        src_output_filepath = f"{dataset_base_path}/df_src.json"
        df_agg = flatten_agg(objects, agg_output_filepath)
        df_src = dedup_per_source(flatten_per_source(objects, src_output_filepath))
        print("df_agg.len: ", len(df_agg))
        print("df_agg[0:1]")
        print(df_agg.head(5))
        print("df_src.len: ", len(df_src))
        print("df_src[0:1]")
        print(df_src.head(5))

        from utils import load_json, write_json
        
        df_agg = load_json(agg_output_filepath)
        df_src = load_json(src_output_filepath)
        df_agg = pd.DataFrame(df_agg)   # if df_agg is currently a list of dicts
        df_src = pd.DataFrame(df_src)   # if df_agg is currently a list of dicts
        
        from dataset_analysis import run_all_analyses
        analyses = run_all_analyses(df_agg, df_src)

        from dataset_analysis_plots import plot_all
        plot_all(analyses, df_agg, df_src) 

    ##########################################################
    ################## Dataset Analysis ends #################
    ##########################################################

    ##########################################################################################################
    ############################################## Translate Test Data starts ################################
    ##########################################################################################################
    # Test Data filepaths 

    catqa_datapath = catqa_dataset_path
    multijail_datapath = multijail_dataset_path
    aegis_datapath = aegis_dataset_path
    toxic_datapath = toxic_dataset_path
    hatespeech_datapath = hatespeech_dataset_path

    # # NLLB-200 3.3B (better quality)
    # # [Done] CatQA + nllb200_3.3b + bn -> sw
    # # 550 data
    if False: # to reduce commented code blocks, doing this check
        stats = translate(
            input_path=catqa_datapath,
            output_path=f"{dataset_base_path}/translation/catqa_nllb200_3.3b_sw.json",
            model_preset="nllb200_3.3b_bn_sw",
            gpu_device=0,
            target_lang_key="sw",
        )
        print(stats)

    # # [DONE] MultiJail + nllb200_3.3b + bn -> sw
    # # 315 data
    if False:
        stats = translate(
            input_path=multijail_datapath,
            output_path=f"{dataset_base_path}/translation/multijail_nllb200_3.3b_sw.json",
            model_preset="nllb200_3.3b_bn_sw",
            gpu_device=0,
            target_lang_key="sw",
        )
        print(stats)

    # # [DONE] Aegis + nllb200_3.3b + bn -> sw
    # # [Issue] translation string empty, 300 data sampled from Aegis, we will do more later
    if False:
        stats = translate(
            input_path=aegis_datapath,
            output_path=f"{dataset_base_path}/translation/aegis_nllb200_3.3b_sw.json",
            model_preset="nllb200_3.3b_bn_sw",
            gpu_device=0,
            target_lang_key="sw",
        )
        print(stats)

    # # [DONE] Toxic + nllb200_3.3b + bn -> sw
    # # 776 data sampled (harm score 1.0) from toxic ground truth, we will do more later
    if False:
        stats = translate(
            input_path=toxic_datapath,
            output_path=f"{dataset_base_path}/translation/toxic_nllb200_3.3b_sw.json",
            model_preset="nllb200_3.3b_bn_sw",
            gpu_device=0,
            target_lang_key="sw",
        )
        print(stats)

    # # [DONE] Hatespeech + nllb200_3.3b + bn -> sw
    # # [Issue] some response empty, 256 data sampled (harm score 1.0) from hatespeech ground truth, we will do more later
    if False:
        stats = translate(
            input_path=hatespeech_datapath,
            output_path=f"{dataset_base_path}/translation/hatespeech_nllb200_3.3b_sw.json",
            model_preset="nllb200_3.3b_bn_sw",
            gpu_device=0,
            target_lang_key="sw",
        )
        print(stats)

    # # bn ------> ha
    # # NLLB-200 3.3B (better quality)
    # # [DONE] CatQA + nllb200_3.3b + bn -> ha
    if False:
        stats = translate(
            input_path=catqa_datapath,
            output_path=f"{dataset_base_path}/translation/catqa_nllb200_3.3b_ha.json",
            model_preset="nllb200_3.3b_bn_ha",
            gpu_device=0,
            target_lang_key="ha",
        )
        print(stats)

    # # [DONE] MultiJail + nllb200_3.3b + bn -> ha
    if False:
        stats = translate(
            input_path=multijail_datapath,
            output_path=f"{dataset_base_path}/translation/multijail_nllb200_3.3b_ha.json",
            model_preset="nllb200_3.3b_bn_ha",
            gpu_device=0,
            target_lang_key="ha",
        )
        print(stats)

    # # [DONE] Aegis + nllb200_3.3b + bn -> sw
    if False:
        stats = translate(
            input_path=aegis_datapath,
            output_path=f"{dataset_base_path}/translation/aegis_nllb200_3.3b_ha.json",
            model_preset="nllb200_3.3b_bn_ha",
            gpu_device=0,
            target_lang_key="ha",
        )
        print(stats)

    # # [DONE] Toxic + nllb200_3.3b + bn -> sw
    if False:
        stats = translate(
            input_path=toxic_datapath,
            output_path=f"{dataset_base_path}/translation/toxic_nllb200_3.3b_ha.json",
            model_preset="nllb200_3.3b_bn_ha",
            gpu_device=0,
            target_lang_key="ha",
        )
        print(stats)

    # # [DONE] Hatespeech + nllb200_3.3b + bn -> sw
    if False:
        stats = translate(
            input_path=hatespeech_datapath,
            output_path=f"{dataset_base_path}/translation/hatespeech_nllb200_3.3b_ha.json",
            model_preset="nllb200_3.3b_bn_ha",
            gpu_device=0,
            target_lang_key="ha",
        )
        print(stats)

    ##########################################################################################################
    ############################################## Translate Test Data ends (will translate all data later) ##################################
    ##########################################################################################################


    ##########################################################################################################
    ############################################## Generate Response starts ##################################
    ##########################################################################################################
    # Family of models: Llama, Qwen, Mistral, Gemma
    # model sizes: will cover at least 2 model sizes from each family of models

    # ************************************************************************
    #************************** Llama Family of models ***********************
    # ************************************************************************
    # CatQA + Llama - done
    # [DONE] CatQA + meta-llama//Meta-Llama-3-8B-Instruct
    if False:
        generate_llama(input_path=catqa_datapath, 
                    output_path=f"{dataset_base_path}/response/catqa_response_llama3_8b_bn.json",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    prompt_key="agg_prompt_bn",
                    id_key="id",
                    system="You are a helpful assistant. Answer in Bangla.")
    # pid=1497545
    
    # MultiJail + Llama - done
    # [DONE] MultiJail + meta-llama//Meta-Llama-3-8B-Instruct
    if False:
        generate_llama(input_path=multijail_datapath, 
                    output_path=f"{dataset_base_path}/response/multijail_response_llama3_8b_bn.json",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    prompt_key="agg_prompt_bn",
                    id_key="id",
                    system="You are a helpful assistant. Answer in Bangla.")
    # pid=1497545
    
    # Aegis + Llama - done
    # [DONE] Aegis + meta-llama/Meta-Llama-3-8B-Instruct
    if False:
        generate_llama(input_path=aegis_datapath, 
                    output_path=f"{dataset_base_path}/response/aegis_response_llama3_8b_bn.json",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    prompt_key="prompt_bn",
                    id_key="id",
                    system="You are a helpful assistant. Answer in Bangla.")
    # llama_aegis.sh
    # pid=1707992
    
    # Toxic + Llama - done
    # [DONE & working but slow, need to filter & reduce data] Toxic + meta-llama//Meta-Llama-3-8B-Instruct
    # filtered with harm_score >= 1.0 & 776 data
    if False:
        generate_llama(input_path=toxic_datapath, 
                    output_path=f"{dataset_base_path}/response/toxic_response_llama3_8b_bn.json",
                    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                    prompt_key="agg_prompt_bn",
                    id_key="id",
                    system="You are a helpful assistant. Answer in Bangla.")
    # llama_toxic.sh
    # pid=2535194

    
    # Hatespeech + Llama - done
    # [DONE] Hatespeech + meta-llama//Meta-Llama-3-8B-Instruct
    # filtered with harm_score >=1.0 & 256 data
    if False:
        generate_llama(input_path=hatespeech_datapath, 
                       output_path=f"{dataset_base_path}/response/hatespeech_response_llama3_8b_bn.json",
                       model_name="meta-llama/Meta-Llama-3-8B-Instruct",
                       prompt_key="agg_prompt_bn",
                       id_key="id",
                       system="You are a helpful assistant. Answer in Bangla.")
    # # llama_hatespeech.sh
    # PID=2819340
    
    
    # ************************************************************************
    #************************** Qwen Family of models ************************
    # ************************************************************************
    # CatQA + Qwen2.5 - done
    # [DONE] CatQA + Qwen/Qwen2.5-7B-Instruct
    if False:
        generate_qwen(
            input_path=catqa_datapath,
            output_path=f"{dataset_base_path}/response/catqa_response_qwen2.5_7b_bn.json",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    # process id 1254033 - running

    # CatQA + Qwen3 - remaining

    
    # MultiJail + Qwen3 - done
    # [DONE] MultiJail + OpenPipe/Qwen3-14B-Instruct
    if False:
        generate_qwen_responses(
            input_path=multijail_datapath,
            output_path=f"{dataset_base_path}/response/multijail_response_qwen3_14b_bn.json",
            model_name="OpenPipe/Qwen3-14B-Instruct",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    # qwen_multijail.sh
    # pid=1609357


    #============> qwen_aegis_toxic_hs.sh together in a single file
    # using this 300 data sampled from Aegis
    # Aegis + Qwen3 - done
    # [DONE] Aegis + OpenPipe/Qwen3-14B-Instruct
    if False:
        generate_responses(
            input_path=aegis_datapath,
            output_path=f"{dataset_base_path}/response/aegis_response_qwen3_14b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    
    # Toxic + Qwen3 - done
    # [DONE] Toxic + OpenPipe/Qwen3-14B-Instruct
    if False:
        generate_responses(
            input_path=toxic_datapath,
            output_path=f"{dataset_base_path}/response/toxic_response_qwen3_14b_bn.json",
            model_name="OpenPipe/Qwen3-14B-Instruct",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    
    # Hatespeech + Qwen3 - done
    # [DONE] Hatespeech + OpenPipe/Qwen3-14B-Instruct
    if False:
        generate_responses(
            input_path=hatespeech_datapath,
            output_path=f"{dataset_base_path}/response/hatespeech_response_qwen3_14b_bn.json",
            model_name="OpenPipe/Qwen3-14B-Instruct",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    #============> qwen_aegis_toxic_hs.sh together in a single file


    # ***************************************************************************
    #************************** Mistral Family of models ************************
    # ***************************************************************************
    # CatQA + Mistral - done
    # [DONE] CatQA + mistralai/Mistral-7B-Instruct-v0.3
    if False:
        generate_mistral_mixtral_faster(
            input_path=catqa_datapath,
            output_path=f"{dataset_base_path}/response/catqa_response_mistral_7b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    # process id 1261698 - running

    # MultiJail + Mistral - done
    # [DONE] MultiJail + mistralai/Mistral-7B-Instruct-v0.3
    
    if False:
        generate_mistral_mixtral_faster(
            input_path=multijail_datapath,
            output_path=f"{dataset_base_path}/response/multijail_response_mistral_7b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )

    
    ############################################################################
    #################### staratified sampling utils starts #####################
    ############################################################################

    # json_path = "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/gt_aegis_formatted.json"
    # data = load_json(json_path)

    # # Count by single category field
    # cnt_severe = count_by_field(data, "most_severe_category", multi=False)

    # # Count by violated categories (can be multi-label)
    # cnt_violated = count_by_field(data, "violated_categories", multi=True)

    # print("Counts (most_severe_category):")
    # for k, v in cnt_severe.most_common():
    #     print(f"{k}: {v}")

    # print("\nCounts (violated_categories, multi-label):")
    # for k, v in cnt_violated.most_common():
    #     print(f"{k}: {v}")

    # input_path = "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/gt_aegis_formatted.json"
    # output_path = "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/gt_aegis_formatted_stratified_samples.json"

    # data = load_json(input_path)

    # sampled, breakdown = stratified_sample_aegis(
    #     data,
    #     category_field="most_severe_category",  # or "violated_categories" if you want multi-label logic
    #     target_n=300,
    #     min_per_category=1,     # set 0 if you don't want tiny categories forced in
    #     max_per_category=None,  # e.g. 60 if you want a cap
    #     seed=42,
    # )

    # write_json(output_path, sampled)

    # print(f"Saved {len(sampled)} samples -> {output_path}")
    # print("Breakdown:")
    # for k, v in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{k}: {v}")

    ############################################################################
    #################### staratified sampling utils ends #######################
    ############################################################################

    # Aegis + Mistral - done
    # [DONE] Aegis + mistralai/Mistral-7B-Instruct-v0.3
    # using this 300 data sampled from Aegis
    if False:
        generate_responses(
            input_path=aegis_datapath,
            output_path=f"{dataset_base_path}/response/aegis_response_mistral_7b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    # mistral_aegis.sh
    # PID=3387332


    # Toxic + Mistral - running
    # [running] Toxic + mistralai/Mistral-7B-Instruct-v0.3
    # filtered with harm_score >=1.0 & 256 data 
    if False:
        generate_mistral_mixtral_faster(
            input_path=toxic_datapath,
            output_path=f"{dataset_base_path}/response/toxic_response_mistral_7b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    # mistral_toxic.sh
    # PID=3831585

    # Hatespeech + Mistral - doing
    # [DONE] Hatespeech + mistralai/Mistral-7B-Instruct-v0.3
    # filtered with harm_score >= 1.0 & 776 data
    if False:
        generate_mistral_mixtral_faster(
            input_path=hatespeech_datapath,
            output_path=f"{dataset_base_path}/response/hatespeech_response_mistral_7b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )
    # mistral_hatespeech.sh
    
    # ***************************************************************************
    #************************** Gemma Family of models **************************
    # ***************************************************************************
    # CatQA + Gemma3 - done
    
    # [DONE] CatQA + google/gemma-3-12b-it
    # input_path = "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/gt_catqa_merged_by_id.json"
    if False:
        generate_gemma_responses(
            input_path=catqa_datapath,
            output_path=f"{dataset_base_path}/response/catqa_response_gemma3_12b_bn.json",
            model_name="google/gemma-3-12b-it",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla.",
        )

    # MultiJail + Gemma3 - done
    # [DONE] MultiJail + google/gemma-3-12b-it
    # input_path = "/home/malam10/projects/ai-safety-bangla/final/data/ground_truth/gt_multijail_merged_by_id.json"
    if False:
        generate_gemma_responses(
            input_path=input_path,
            output_path=f"{dataset_base_path}/response/multijail_response_gemma3_12b_bn.json",
            model_name="google/gemma-3-12b-it",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla.",
        )

    # Aegis + Mistral - done
    # [DONE] Aegis + mistralai/Mistral-7B-Instruct-v0.3
    # using this 300 data sampled from Aegis
    if False:
        generate_responses(
            input_path=aegis_datapath,
            output_path=f"{dataset_base_path}/response/aegis_response_gemma3_12b_bn.json",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            prompt_key="prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla."
        )

    # Hatespeech + Gemma3  - done
    # [DONE] Hatespeech + google/gemma-3-12b-it
    # filtered with harm_score >= 1.0 & 776 data
    if False:
        generate_gemma_responses(
            input_path=hatespeech_datapath,
            output_path=f"{dataset_base_path}/response/hatespeech_response_gemma3_12b_bn.json",
            model_name="google/gemma-3-12b-it",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla.",
        )
    # gemma_toxic.sh - actually hatespeech
    # PID=2931887

    # Toxic + Gemma3  - done
    # [DONE] Toxic + google/gemma-3-12b-it
    # filtered with harm_score >=1.0 & 256 data 
    if False:
        generate_gemma_responses(
            input_path=toxic_datapath,
            output_path=f"{dataset_base_path}/response/toxic_response_gemma3_12b_bn.json",
            model_name="google/gemma-3-12b-it",
            prompt_key="agg_prompt_bn",
            id_key="id",
            system="You are a helpful assistant. Answer in Bangla.",
        )
    # gemma_hatespeech.sh - actually toxic

    ##########################################################################################################
    ############################################## Generate Response ends ####################################
    ##########################################################################################################


    ##########################################################################################################
    ############################################ Evaluation of Generated Responses starts ####################
    ##########################################################################################################

    from evaluation import load_family_dataset_file, records_to_df, compute_all_results, make_all_plots, latex_table_main_results
    # Example: load a few files (edit paths)
    records = []
    # Llama Family 
    records += load_family_dataset_file(f"{dataset_base_path}/response/catqa_response_llama3_8b_bn.json", dataset="CatQA", family="Llama")
    records += load_family_dataset_file(f"{dataset_base_path}/response/multijail_response_llama3_8b_bn.json", dataset="MultiJail", family="Llama")
    records += load_family_dataset_file(f"{dataset_base_path}/response/aegis_response_llama3_8b_bn.json", dataset="Aegis", family="Llama")
    records += load_family_dataset_file(f"{dataset_base_path}/response/toxic_response_llama3_8b_bn.json", dataset="Toxic", family="Llama")
    records += load_family_dataset_file(f"{dataset_base_path}/response/hatespeech_response_llama3_8b_bn.json", dataset="Hatespeech", family="Llama")
    print("After loading Llama data records.len: ", len(records))

    # Qwen Family
    # catqa + Qwen3 remaining 
    records += load_family_dataset_file(f"{dataset_base_path}/response/catqa_response_qwen2.5_7b_bn.json", dataset="CatQA", family="Qwen")
    records += load_family_dataset_file(f"{dataset_base_path}/response/multijail_response_qwen3_14b_bn.json", dataset="MultiJail", family="Qwen")
    records += load_family_dataset_file(f"{dataset_base_path}/response/aegis_response_qwen3_14b_bn.json", dataset="Aegis", family="Qwen")
    records += load_family_dataset_file(f"{dataset_base_path}/response/toxic_response_qwen3_14b_bn.json", dataset="Toxic", family="Qwen")
    records += load_family_dataset_file(f"{dataset_base_path}/response/hatespeech_response_qwen3_14b_bn.json", dataset="Hatespeech", family="Qwen")
    print("After loading Qwen data records.len: ", len(records))

    # Mistral Family
    records += load_family_dataset_file(f"{dataset_base_path}/response/catqa_response_mistral_7b_bn.json", dataset="CatQA", family="Mistral")
    records += load_family_dataset_file(f"{dataset_base_path}/response/multijail_response_mistral_7b_bn.json", dataset="MultiJail", family="Mistral")
    records += load_family_dataset_file(f"{dataset_base_path}/response/aegis_response_mistral_7b_bn.json", dataset="Aegis", family="Mistral")
    records += load_family_dataset_file(f"{dataset_base_path}/response/toxic_response_mistral_7b_bn.json", dataset="Toxic", family="Mistral")
    records += load_family_dataset_file(f"{dataset_base_path}/response/hatespeech_response_mistral_7b_bn.json", dataset="Hatespeech", family="Mistral")
    print("After loading Mistral data records.len: ", len(records))

    # Gemma Family
    records += load_family_dataset_file(f"{dataset_base_path}/response/catqa_response_gemma3_12b_bn.json", dataset="CatQA", family="Gemma")
    records += load_family_dataset_file(f"{dataset_base_path}/response/multijail_response_gemma3_12b_bn.json", dataset="MultiJail", family="Gemma")
    records += load_family_dataset_file(f"{dataset_base_path}/response/aegis_response_gemma3_12b_bn.json", dataset="Aegis", family="Gemma")
    records += load_family_dataset_file(f"{dataset_base_path}/response/toxic_response_gemma3_12b_bn.json", dataset="Toxic", family="Gemma")
    records += load_family_dataset_file(f"{dataset_base_path}/response/hatespeech_response_gemma3_12b_bn.json", dataset="Hatespeech", family="Gemma")
    print("After loading Gemma data records.len: ", len(records))
    
    
    print("records.len: ", len(records))
    
    if False:
        df = records_to_df(records)
        
        results = compute_all_results(df)
        for k, v in results.items():
            print("\n", k)
            print(v.head())
        
        make_all_plots(df)
    
        # print(latex_table_main_results())

    ##########################################################################################################
    ################################ Evaluation of Generated Responses ends ##################################
    ##########################################################################################################

    # remaining tasks 
    # 1) Translation: translation into other languages (select the languages first - around 10 languages from different language families and with different resource levels, e.g. Hindi, Swahili, Hausa, Yoruba, etc.) - [in progress]
    # -- several iterations for improving translation quality (e.g. prompt engineering, model selection, post-processing, etc.)
    # 2) Response Generation: response generation for each languages using 4 family of models (Llama, Qwen, Mistral, Gemma) - [in progress]
    # 3) Evaluation: evaluation of the responses using both human evaluation and automatic metrics (e.g. BERTScore, BLEURT, etc.) - [not started]
    # 4) Analysis: analysis and visualization of the results - [not started]
    # 5) Jailbreak generation techniques like AutoDAN, DAN, etc. - [not started]










