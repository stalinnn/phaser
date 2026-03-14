import json
import random

def generate_hierarchy_data():
    """
    Generate a scaled-up synthetic dataset simulating the hierarchical structure of a complex novel (e.g., Dream of the Red Chamber) 
    or a corporate financial structure.
    We need positive pairs (parent-child causality or macro-micro abstraction) and negative pairs (unrelated).
    """
    
    # 1. Define the Macro/Root concepts
    macros = [
        "贾府的经济状况逐渐衰退，入不敷出。",
        "大观园内实行严格的封建等级制度。",
        "王熙凤掌管荣国府内务，手段严厉。",
        "贾宝玉与林黛玉之间存在着深刻的木石前盟爱情悲剧。",
        "贾政对贾宝玉的教育极其严苛，期望他走仕途经济之路。"
    ]
    
    # 2. Define Micro/Leaf concepts that logically follow from Macro
    micro_dict = {
        macros[0]: [
            "今天贾母过寿，厨房为了节省开支，偷偷把燕窝换成了普通的银耳。",
            "探春发现月钱已经两个月没有按时发放了，心里十分焦急。",
            "为了填补亏空，贾琏悄悄把老太太的一件古董花瓶拿去当铺死当了。",
            "过年时的赏赐比往年减少了一半，底下的丫鬟们都在私下抱怨。"
        ],
        macros[1]: [
            "因为一个丫鬟打碎了茶碗，被管家直接拉出去打板子，甚至要撵出府去。",
            "赵姨娘因为庶出的身份，在分发布料时总是分到最差的挑剩下的。",
            "主子们吃饭时，即使是贴身大丫鬟也只能站在旁边伺候，绝不能上桌。",
            "门子拦住了来打秋风的刘姥姥，只因为她衣衫褴褛，不符合府里的规矩。"
        ],
        macros[2]: [
            "凤姐查出尤氏房里的丫头偷懒，当众羞辱并扣了她三个月的月钱。",
            "为了树立威信，王熙凤在协理宁国府时，对迟到一刻钟的仆人严惩不贷。",
            "平儿虽然受宠，但在凤姐面前依然战战兢兢，不敢有丝毫越矩。",
            "下人们私下里都叫她‘夜叉’，对她下达的命令总是阳奉阴违又不敢抗拒。"
        ],
        macros[3]: [
            "黛玉看到宝玉把别人送的玉佩戴在身上，伤心地把自己的香囊剪成了两半。",
            "宝玉在梦里喊着‘和尚道士的话如何信得？什么是金玉良缘？我偏说是木石前盟！’",
            "林妹妹因为宝玉多和宝钗说了几句话，便独自一人在花阴下哭泣。",
            "宝玉发呆时总觉得这个妹妹似乎在哪里见过，有一种难以言喻的熟悉感。"
        ],
        macros[4]: [
            "贾政突然查问宝玉的功课，发现他连《四书》都没背熟，大发雷霆。",
            "宝玉因为害怕父亲的责骂，常常躲在姐妹们的房间里不敢出去。",
            "贾政怒骂宝玉‘不务正业，专在内帏厮混’，气得拿起了大板子。",
            "清客相公们为了迎合贾政，也总是劝宝玉多读些制艺文章，早日考取功名。"
        ]
    }
    
    dataset = []
    
    # Generate Positive Pairs (Label = 1)
    for macro, micros in micro_dict.items():
        for micro in micros:
            dataset.append({
                "parent": macro,
                "child": micro,
                "label": 1
            })
            
    # Generate Negative Pairs (Label = -1)
    # Mix and match across different macros to create logical disconnects
    all_micros = [m for micros in micro_dict.values() for m in micros]
    
    for macro in macros:
        # Get micros that do NOT belong to this macro
        unrelated_micros = [m for m in all_micros if m not in micro_dict[macro]]
        # Sample a few to create negative pairs
        sampled_negatives = random.sample(unrelated_micros, k=4)
        for neg_micro in sampled_negatives:
            dataset.append({
                "parent": macro,
                "child": neg_micro,
                "label": -1
            })
            
    # Add some hard negatives (lexically similar but logically unrelated)
    # E.g., both contain "贾宝玉" but reason is different
    hard_negatives = [
        {"parent": "贾政对贾宝玉的教育极其严苛，期望他走仕途经济之路。", "child": "宝玉在梦里喊着什么是金玉良缘，我偏说是木石前盟！", "label": -1},
        {"parent": "贾宝玉与林黛玉之间存在着深刻的木石前盟爱情悲剧。", "child": "贾政突然查问宝玉的功课，发现他连《四书》都没背熟，大发雷霆。", "label": -1},
        {"parent": "凤姐查出尤氏房里的丫头偷懒，当众羞辱并扣了她三个月的月钱。", "child": "为了填补亏空，贾琏悄悄把老太太的一件古董花瓶拿去当铺死当了。", "label": -1}
    ]
    
    dataset.extend(hard_negatives)
    
    random.shuffle(dataset)
    
    with open("hierarchical_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Generated {len(dataset)} hierarchical pairs (Positives and Negatives).")
    print("Dataset saved to hierarchical_dataset.json")

if __name__ == "__main__":
    generate_hierarchy_data()
