"""
Run Full Evaluation Script
Bilingual Document Q&A (RAG) System

Executes:
1. Retrieval evaluation (Baseline vs Reranked)
2. Generation quality evaluation (Correctness, Unsupported rate, Refusal accuracy)
Outputs summary tables and writes results to the evaluation/ directory.
"""

import os
import sys
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    print("=" * 70)
    print("STEP 1: RUNNING RETRIEVAL EVALUATION")
    print("=" * 70)
    ret_script = os.path.join("evaluation", "evaluate_retrieval.py")
    ret_code = subprocess.call([sys.executable, ret_script])
    if ret_code != 0:
        print(f"Error: Retrieval evaluation exited with code {ret_code}")
        sys.exit(ret_code)

    print("\n" + "=" * 70)
    print("STEP 2: RUNNING ANSWER EVALUATION")
    print("=" * 70)
    ans_script = os.path.join("evaluation", "evaluate_answers.py")
    ans_code = subprocess.call([sys.executable, ans_script])
    if ans_code != 0:
        print(f"Error: Answer evaluation exited with code {ans_code}")
        sys.exit(ans_code)

    print("\n" + "=" * 70)
    print("FULL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
