# extensions/run_all_attacks.py
import sys
import os
import argparse
import subprocess
import time
from datetime import datetime

def run_attack(script_name: str, target_model: str, dataset: str, iterations: int, max_questions: int):
    """Run a single attack script."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    cmd = [
        "python",
        f"extensions/{script_name}",
        "--target_model", target_model,
        "--dataset", dataset,
        "--iterations", str(iterations),
        "--max_questions", str(max_questions)
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Completed in {end_time - start_time:.2f} seconds")
    return result.returncode

def run_all_attacks(target_model: str, dataset: str, iterations: int, max_questions: int):
    """Run all attack scripts in sequence."""
    attacks = [
        "attack_dynamic_mutation.py",
        "attack_cot_poisoning.py",
        "attack_multiturn.py",
        "attack_ensemble.py",
        "attack_embedding_fitness.py",
        "attack_multilingual.py"
    ]
    
    print(f"\n{'='*60}")
    print(f"RUNNING ALL ATTACKS")
    print(f"Target Model: {target_model}")
    print(f"Dataset: {dataset}")
    print(f"Iterations per attack: {iterations}")
    print(f"Max questions: {max_questions}")
    print(f"{'='*60}")
    
    total_start = time.time()
    results = {}
    
    for attack in attacks:
        attack_name = attack.replace("attack_", "").replace(".py", "")
        return_code = run_attack(attack, target_model, dataset, iterations, max_questions)
        results[attack_name] = "Success" if return_code == 0 else "Failed"
    
    total_end = time.time()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL ATTACKS")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name}: {status}")
    print(f"\nTotal execution time: {total_end - total_start:.2f} seconds")
    print(f"\nAll results have been saved to the database.")
    print("Run 'python extensions/dashboard_builder.py' to view the dashboard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all attack scripts")
    parser.add_argument("--target_model", type=str, required=True,
                       help="Target model name")
    parser.add_argument("--dataset", type=str,
                       default="./datasets/questions/question_target.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--iterations", type=int, default=20,
                       help="Number of iterations per question per attack")
    parser.add_argument("--max_questions", type=int, default=20,
                       help="Maximum number of questions to process")
    
    args = parser.parse_args()
    run_all_attacks(args.target_model, args.dataset, args.iterations, args.max_questions)