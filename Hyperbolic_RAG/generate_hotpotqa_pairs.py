import json
import random
import os
from tqdm import tqdm

def generate_hotpotqa_training_data(input_file, output_file, max_samples=5000):
    """
    Extract (Question, Context, Label) pairs from HotpotQA dataset for fine-tuning.
    We treat the Question as the 'Parent/Macro' node (closer to origin)
    and the supporting facts/contexts as 'Child/Micro' nodes.
    """
    print(f"Loading HotpotQA data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    dataset = []
    
    # Shuffle data to get a random sample
    random.shuffle(data)
    
    count = 0
    print("Generating training pairs...")
    for item in tqdm(data):
        if count >= max_samples:
            break
            
        question = item['question']
        
        # In HotpotQA, supporting_facts is a list of [title, sentence_index]
        supporting_titles = set([fact[0] for fact in item['supporting_facts']])
        
        # Context is a list of [title, [sentence1, sentence2, ...]]
        contexts = item['context']
        
        positive_contexts = []
        negative_contexts = []
        
        for title, sentences in contexts:
            # Join sentences to form a chunk
            chunk = f"Title: {title}. " + " ".join(sentences)
            if title in supporting_titles:
                positive_contexts.append(chunk)
            else:
                negative_contexts.append(chunk)
                
        # Generate positive pairs (Label = 1)
        for pos_ctx in positive_contexts:
            dataset.append({
                "parent": question,
                "child": pos_ctx,
                "label": 1
            })
            
        # Generate negative pairs (Label = -1) from distractors within the SAME question
        # This makes them "hard negatives" because they are topically related
        # but not the actual supporting facts needed to answer the question.
        # We sample up to 2 hard negatives per question to balance the dataset.
        sampled_negatives = random.sample(negative_contexts, min(len(negative_contexts), 2))
        for neg_ctx in sampled_negatives:
            dataset.append({
                "parent": question,
                "child": neg_ctx,
                "label": -1
            })
            
        count += 1
        
    print(f"\nGenerated {len(dataset)} training pairs (Positives and Hard Negatives).")
    
    # Shuffle the final dataset
    random.shuffle(dataset)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Training dataset saved to {output_file}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "datasets", "HotpotQA", "hotpot_train_v1.1.json")
    output_file = os.path.join(base_dir, "Scale_Up_Holo_RAG", "hotpotqa_train_pairs.json")
    
    # Generate 5000 questions worth of pairs (approx 15000-20000 pairs total)
    generate_hotpotqa_training_data(input_file, output_file, max_samples=5000)
