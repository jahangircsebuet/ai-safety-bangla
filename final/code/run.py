import sys
from setproctitle import setproctitle
setproctitle("run.py")

def run_single_pass_llama2_aegis():
    sys.argv = [
        "working_code_6_categorize_score_using_open_llm_llama2_7b_chat.py",
        "--input", "/home/malam10/projects/ai-safety-bangla/final/data/converted_aegis_v2_en_bn.json",
        "--output", "/home/malam10/projects/ai-safety-bangla/final/data/converted_aegis_v2_en_bn_open_llm_llama2_7b_chat_single_run.json",
        "--k", "1",
    ]

    from working_code_6_categorize_score_using_open_llm_llama2_7b_chat import main
    main()

# IMPORTANT: import AFTER setting argv
def run_single_pass_llama3():
    sys.argv = [
        "working_code_6_categorize_score_using_open_llm_llama3_8b_instruct_copy.py",
        "--input", "/home/malam10/projects/ai-safety-bangla/final/data/converted_aegis_v2_en_bn.json",
        "--output", "/home/malam10/projects/ai-safety-bangla/final/data/converted_aegis_v2_en_bn_open_llm_llama3_8b_instruct_single_run.json",
        "--k", "1",
    ]

    from working_code_6_categorize_score_using_open_llm_llama3_8b_instruct_copy import main
    main()


if __name__ == "__main__":
    # run_single_pass_llama3()
    run_single_pass_llama2_aegis()
