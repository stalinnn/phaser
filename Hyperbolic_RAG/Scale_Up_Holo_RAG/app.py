import gradio as gr
import torch
import json
import os
import sys

# Import the core components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from holo_embedder import HyperbolicEmbedder
from run_holo_rag import FlatVectorDB
from batched_holo_db import BatchedHoloVectorDB

print("初始化 Holo-RAG Demo 引擎，请稍候...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models globally so they don't reload on every button click
try:
    embedder = HyperbolicEmbedder("BAAI/bge-small-zh-v1.5").to(device)
    # Load the highly penalized fine-tuned weights! This is crucial for the demo effect.
    embedder.projection.load_state_dict(torch.load("scale_holo_projection.pt", map_location=device))
    embedder.eval()
    print("模型加载成功。")
except Exception as e:
    print(f"模型加载失败: {e}")
    sys.exit(1)

# Pre-load the Red Chamber dataset for the demo
db_flat = FlatVectorDB(device=device)
db_holo = BatchedHoloVectorDB(c=1.0, device=device)

with open("hierarchical_dataset.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
    
kb_set = set()
for item in data:
    kb_set.add(item['parent'])
    kb_set.add(item['child'])
knowledge_base = list(kb_set)

print(f"正在将 {len(knowledge_base)} 条知识同时存入欧氏空间与双曲空间...")
batch_size = 16
with torch.no_grad():
    for i in range(0, len(knowledge_base), batch_size):
        batch_texts = knowledge_base[i:i+batch_size]
        inputs = embedder.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        euclidean_embs = outputs.last_hidden_state[:, 0, :]
        scaled_embs = embedder.projection(euclidean_embs)
        hyperbolic_embs = embedder.holo_math.exp_map0(scaled_embs)
        
        db_flat.add_texts(euclidean_embs, batch_texts)
        db_holo.add_texts(hyperbolic_embs, batch_texts)
print("知识库就绪！")

def process_query(query):
    """
    Given a query, search both databases and format the output.
    """
    if not query.strip():
        return "请输入问题！", "请输入问题！"
        
    inputs = embedder.tokenizer([query], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = embedder.backbone(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        q_euc = outputs.last_hidden_state[:, 0, :]
        q_hyp = embedder.holo_math.exp_map0(embedder.projection(q_euc))
        
    flat_results = db_flat.search(q_euc, top_k=3)
    holo_results = db_holo.search(q_hyp, top_k=3)
    
    # Format Flat Results
    flat_html = "<div style='padding: 10px; background-color: #f8d7da; border-radius: 5px;'>"
    flat_html += "<h3 style='color: #721c24; margin-top: 0;'>⚠️ 仅字面匹配 (Cosine Similarity)</h3>"
    for rank, (score, text) in enumerate(flat_results, 1):
        flat_html += f"<p><b>Top {rank}</b> (相似度 {score:.2f}):<br><i>{text}</i></p>"
    flat_html += "</div>"
    
    # Format Holo Results
    holo_html = "<div style='padding: 10px; background-color: #d4edda; border-radius: 5px;'>"
    holo_html += "<h3 style='color: #155724; margin-top: 0;'>🎯 隐式逻辑召回 (Poincaré Geodesic)</h3>"
    for rank, (dist, text) in enumerate(holo_results, 1):
        holo_html += f"<p><b>Top {rank}</b> (双曲距离 {dist:.2f}):<br><b>{text}</b></p>"
    holo_html += "</div>"
    
    return flat_html, holo_html

# --- Gradio UI Definition ---
with gr.Blocks(title="Holo-RAG: 全息检索增强生成", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌌 Holo-RAG: 全息双曲检索演示 (降维打击版)
        **告别“字面匹配”，无需构建昂贵知识图谱，通过数学流形测地线“一笔画”召回深层逻辑！**
        
        目前内置了**《红楼梦》因果逻辑测试集**。请在下方输入您的**多跳因果问题**，对比传统平直空间（左）与双曲流形空间（右）的召回结果差异。
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                lines=2, 
                placeholder="请输入诸如：'刘姥姥被门子拦住的深层原因是什么？' 等需要逻辑推理的问题...", 
                label="用户提问 (Query)"
            )
            
            with gr.Row():
                clear_btn = gr.Button("清空")
                submit_btn = gr.Button("🚀 弯曲测地线检索", variant="primary")
            
            gr.Examples(
                examples=[
                    "刘姥姥被门子拦住的深层原因是什么？",
                    "探春为什么因为月钱没发而焦急？",
                    "平儿在凤姐面前战战兢兢说明了什么？",
                    "宝玉害怕父亲责骂躲在姐妹房间的根源？",
                    "黛玉看到宝玉戴别人的玉佩为什么剪香囊？"
                ],
                inputs=query_input,
                label="试试这些极其困难的隐式推理问题："
            )
            
        with gr.Column(scale=3):
            gr.Image(value="../Holo_RAG_Paper/poincare_visualization.png", label="特征空间几何透视 (庞加莱圆盘)", show_label=True, interactive=False)
            
    gr.Markdown("### 🔍 召回结果实时对比")
    with gr.Row():
        flat_output = gr.HTML(label="传统 RAG (平直空间)")
        holo_output = gr.HTML(label="Holo-RAG (双曲空间)")
        
    submit_btn.click(fn=process_query, inputs=query_input, outputs=[flat_output, holo_output])
    clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[query_input, flat_output, holo_output])
    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
