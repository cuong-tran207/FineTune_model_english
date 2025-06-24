import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss, pickle
import numpy as np

# Load models once at import
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("output")
model = AutoModelForCausalLM.from_pretrained("output", device_map="auto", torch_dtype=torch.bfloat16)

# Load FAISS index & docs
index = faiss.read_index("embeddings/vector_index.faiss")
with open("embeddings/docs.pkl", "rb") as f:
    documents = pickle.load(f)

system_message = """Bạn là một trợ lý thông minh, hãy trở lời câu hỏi hiện tại của user dựa trên lịch sử chat và các tài liệu liên quan.
Câu trả lời phải ngắn gọn, chính xác nhưng vẫn đảm bảo đầy đủ các ý chính.
NOTE: - Hãy chỉ trả lời nếu câu trả lời nằm trong tài liệu được truy xuất ra.
      - Nếu không tìm thấy câu trả lời trong tài liệu truy xuất ra thì hãy trả về : "no" .
Context: {context}"""

def retrieve_context(query, top_k=3):
    emb = embedder.encode([query])
    D, I = index.search(np.array(emb), top_k)
    return "\n".join([documents[i] for i in I[0]])

def build_prompt(context, question):
    return [
        {"role": "system", "content": system_message.format(context=context)},
        {"role": "user", "content": question}
    ]

def chat(messages):
    from trl import ChatFormatter  # Ensure installed
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_answer(question: str) -> str:
    context = retrieve_context(question)
    prompt = build_prompt(context, question)
    print(chat(prompt))  # Debugging output
    return chat(prompt)
