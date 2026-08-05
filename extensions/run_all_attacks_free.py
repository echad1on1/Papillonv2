# extensions/run_all_attacks_free.py
import sys
import os
import argparse
import subprocess
import time
from datetime import datetime

def setup_free_apis():
    """Guide user through setting up free APIs"""
    print("\n" + "="*60)
    print("FREE API SETUP GUIDE")
    print("="*60)
    
    print("\nOption 1: Google Gemini (Recommended)")
    print("  1. Go to: https://makersuite.google.com/app/apikey")
    print("  2. Sign in with your Google account")
    print("  3. Click 'Create API Key'")
    print("  4. Copy the key and set it as environment variable:")
    print("     export GEMINI_API_KEY='your-key-here'")
    print("     OR create config/gemini_key.txt with just the key")
    
    print("\nOption 2: Hugging Face Inference API")
    print("  1. Go to: https://huggingface.co/settings/tokens")
    print("  2. Create a new token (free)")
    print("  3. Set it as environment variable:")
    print("     export HF_API_KEY='your-token-here'")
    print("     OR create config/hf_key.txt with just the token")
    
    print("\nOption 3: Mock Server (No setup required)")
    print("  Use --mock flag for deterministic testing")
    print("  No API key needed")
    print("\n" + "="*60)

def run_attack_free(script_name: str, target_type: str, target_model: str, 
                    dataset: str, iterations: int, max_questions: int, use_mock: bool):
    """Run a single attack script."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")
    
    cmd = [
        "python",
        f"extensions/{script_name}",
        "--target_type", target_type,
        "--target_model", target_model,
        "--dataset", dataset,
        "--iterations", str(iterations),
        "--max_questions", str(max_questions)
    ]
    
    if use_mock:
        cmd.append("--mock")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Completed in {end_time - start_time:.2f} seconds")
    return result.returncode

def run_all_attacks_free(target_type: str, target_model: str, dataset: str, 
                         iterations: int, max_questions: int, use_mock: bool):
    """Run all 6 attack scripts with free APIs."""
    
    print(f"\n{'='*60}")
    print(f"RUNNING ALL 6 ATTACKS (FREE VERSION)")
    print(f"{'='*60}")
    print(f"Target Type: {target_type}")
    print(f"Target Model: {target_model}")
    print(f"Iterations per attack: {iterations}")
    print(f"Max questions: {max_questions}")
    print(f"Using Mock: {use_mock}")
    
    attacks = [
        "attack_dynamic_mutation_free.py",
        "attack_cot_poisoning_free.py",
        "attack_multiturn_free.py",
        "attack_ensemble_free.py",
        "attack_embedding_fitness_free.py",
        "attack_multilingual_free.py"
    ]
    
    total_start = time.time()
    results = {}
    
    for attack in attacks:
        attack_name = attack.replace("attack_", "").replace("_free.py", "")
        return_code = run_attack_free(
            attack, target_type, target_model, 
            dataset, iterations, max_questions, use_mock
        )
        results[attack_name] = "Success" if return_code == 0 else "Failed"
    
    total_end = time.time()
    
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL ATTACKS")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{name}: {status}")
    print(f"\nTotal execution time: {total_end - total_start:.2f} seconds")
    print(f"\nAll results have been saved to the database.")
    print("Run 'python extensions/dashboard_builder_free.py' to view the dashboard.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all 6 attacks (FREE version)")
    parser.add_argument("--target_type", type=str, default="gemini",
                       choices=["gemini", "huggingface", "mock"],
                       help="Type of free LLM to use")
    parser.add_argument("--target_model", type=str, default="gemini-pro",
                       help="Model name")
    parser.add_argument("--dataset", type=str,
                       default="./datasets/questions/question_target.csv",
                       help="Path to dataset CSV")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations per question per attack")
    parser.add_argument("--max_questions", type=int, default=10,
                       help="Maximum number of questions to process")
    parser.add_argument("--mock", action="store_true",
                       help="Use mock server for testing")
    parser.add_argument("--setup", action="store_true",
                       help="Show setup guide for free APIs")
    
    args = parser.parse_args()
    
    if args.setup:
        setup_free_apis()
    else:
        run_all_attacks_free(
            args.target_type,
            args.target_model,
            args.dataset,
            args.iterations,
            args.max_questions,
            args.mock
        )