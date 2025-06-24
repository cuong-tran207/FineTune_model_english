import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_process_data(input_file):
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            lesson = json.loads(line)
            for qa in lesson.get("qa_pairs", []):
                data.append({
                    "question": qa.get("question", ""),
                    "context": lesson.get("explanation", ""),
                    "answer": qa.get("answer", "")
                })
    return pd.DataFrame(data)

def main():
    os.makedirs("data", exist_ok=True)
    input_file = "data/english_lessons_100.jsonl"
    train_output = "data/train.csv"
    test_output = "data/test.csv"
    print("Loading and processing data...")
    df = load_and_process_data(input_file)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2,  # 80% train, 20% test
        random_state=42,
        shuffle=True
    )
    
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"Data split complete!")
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    print(f"Train data saved to: {os.path.abspath(train_output)}")
    print(f"Test data saved to: {os.path.abspath(test_output)}")

if __name__ == "__main__":
    main()