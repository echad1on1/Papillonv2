# extensions/common_utils.py
import sys
import os
import json
import sqlite3
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import random
import hashlib

# Add the project root to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing Papillon components
from Judge.language_models import GPT, Claude, Gemini, LanguageModel
from Judge.judge import Judge

# Database configuration
DB_PATH = "./results/extensions_results.db"

def get_judge(model_type: str = "openai", model_name: str = "gpt-3.5-turbo") -> LanguageModel:
    """Factory function to get the appropriate LLM judge."""
    if model_type == "openai":
        return GPT(model_name)
    elif model_type == "anthropic":
        return Claude(model_name)
    elif model_type == "google":
        return Gemini(model_name)
    else:
        raise ValueError(f"Unsupported judge model: {model_type}")

def load_questions_from_dataset(dataset_path: str, max_questions: Optional[int] = None) -> List[Dict[str, str]]:
    """Loads questions from a Papillon dataset CSV."""
    try:
        df = pd.read_csv(dataset_path)
        # Try to find the question column (different datasets might have different column names)
        question_col = None
        for col in ['question', 'Question', 'prompt', 'Prompt', 'text', 'Text']:
            if col in df.columns:
                question_col = col
                break
        
        if question_col is None:
            raise ValueError(f"No question column found in {dataset_path}")
        
        questions = df[question_col].tolist()
        
        # Also get targets if available
        target_col = None
        for col in ['target', 'Target', 'goal', 'Goal']:
            if col in df.columns:
                target_col = col
                break
        
        targets = df[target_col].tolist() if target_col else [None] * len(questions)
        
        # Create a list of question dictionaries
        result = []
        for q, t in zip(questions, targets):
            if pd.notna(q):
                result.append({
                    'question': str(q),
                    'target': str(t) if pd.notna(t) else None
                })
        
        if max_questions:
            result = result[:max_questions]
            
        return result
        
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        # Fallback to a simple question list
        return [{'question': "How to hack a computer?", 'target': None}]

def save_attack_results(attack_name: str, results: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
    """Saves results to a central SQLite database and JSON file."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create tables
        c.execute('''CREATE TABLE IF NOT EXISTS attacks
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      attack_name TEXT,
                      timestamp TEXT,
                      question TEXT,
                      mutation TEXT,
                      response TEXT,
                      roberta_score REAL,
                      llm_judge_decision TEXT,
                      success INTEGER,
                      metadata TEXT)''')
        
        # Create table for tracking strategy performance
        c.execute('''CREATE TABLE IF NOT EXISTS strategy_performance
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      attack_name TEXT,
                      strategy_name TEXT,
                      success_count INTEGER,
                      total_count INTEGER,
                      success_rate REAL,
                      timestamp TEXT)''')
        
        timestamp = datetime.now().isoformat()
        for result in results:
            c.execute("""INSERT INTO attacks 
                         (attack_name, timestamp, question, mutation, response, 
                          roberta_score, llm_judge_decision, success, metadata) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                      (attack_name, timestamp, 
                       result.get('question', ''), 
                       result.get('mutation', ''),
                       result.get('response', ''),
                       result.get('roberta_score', 0.0),
                       result.get('llm_judge_decision', ''),
                       int(result.get('success', False)),
                       json.dumps(metadata) if metadata else '{}'))
        
        conn.commit()
        conn.close()
        
        # Also save to a JSON file for easy access
        json_path = f"./results/{attack_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({
                'attack_name': attack_name,
                'timestamp': timestamp,
                'metadata': metadata,
                'results': results
            }, f, indent=2)
            
        print(f"Results saved to {json_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def load_roberta_judge():
    """Load the first-stage RoBERTa judge."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        
        model_path = "./roberta"
        if not os.path.exists(model_path):
            print("Warning: RoBERTa judge not found at ./roberta")
            return None
            
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        
        return {
            'model': model,
            'tokenizer': tokenizer
        }
    except Exception as e:
        print(f"Error loading RoBERTa judge: {e}")
        return None

def evaluate_response_with_roberta(response: str, roberta_judge: Any) -> float:
    """Evaluate a response using the RoBERTa judge."""
    if roberta_judge is None:
        return 0.5  # Default score
        
    try:
        from transformers import pipeline
        # Simplified evaluation - in practice you'd use the proper inference
        inputs = roberta_judge['tokenizer'](response, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = roberta_judge['model'](**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            return scores[0][1].item()  # Probability of being harmful
    except Exception as e:
        print(f"Error evaluating with RoBERTa: {e}")
        return 0.5

def create_hash(text: str) -> str:
    """Create a hash of text for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()

class MutationStrategy:
    """Base class for mutation strategies."""
    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.total_count = 0
        
    def get_success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    def mutate(self, question: str, **kwargs) -> str:
        """Apply the mutation strategy to the question."""
        raise NotImplementedError
    
    def update(self, success: bool):
        self.total_count += 1
        if success:
            self.success_count += 1