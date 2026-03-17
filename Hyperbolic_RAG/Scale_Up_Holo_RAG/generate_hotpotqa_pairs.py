import json
import random
import os

def generate_pairs(input_file, output_file, max_samples=20000):
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pairs = []
    
    # Process data to generate positive and negative pairs
    for item in data:
        question = item['question']
        supporting_facts = item['supporting_facts'] # list of [title, index]
        
        # Build lookup for supporting sentences
        sup_dict = {}
        for title, idx in supporting_facts:
            if title not in sup_dict:
                sup_dict[title] = set()
            sup_dict[title].add(idx)
            
        contexts = item['context'] # list of [title, list_of_sentences]
        
        for title, sentences in contexts:
            for idx, sentence in enumerate(sentences):
                # Positive pair
                if title in sup_dict and idx in sup_dict[title]:
                    pairs.append({
                        "parent": question,
                        "child": sentence,
                        "label": 1
                    })
                # Negative pair
                else:
                    # Randomly sample some negative pairs to balance
                    if random.random() < 0.1:
                        pairs.append({
                            "parent": question,
                            "child": sentence,
                            "label": -1
                        })
                        
        if len(pairs) >= max_samples:
            break
            
    # Truncate and shuffle
    pairs = pairs[:max_samples]
    random.shuffle(pairs)
    
    print(f"Generated {len(pairs)} pairs. Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    input_path = "/gz-data/datasets/HotpotQA/hotpot_train_v1.1.json"
    output_path = "hotpotqa_train_pairs.json"
    generate_pairs(input_path, output_path)
