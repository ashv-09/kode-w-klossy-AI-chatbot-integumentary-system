import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# 1. Load knowledge base files safely
file_paths = ["skin.txt", "nails.txt", "hair.txt"]
knowledge_base = ""

for path in file_paths:
    try:
        with open(path, "r", encoding="utf-8") as file:
            knowledge_base += file.read() + "\n\n"
    except FileNotFoundError:
        print(f"Warning: {path} was not found.")

print(f"Loaded knowledge base length: {len(knowledge_base)} characters.")


# 2. Preprocess text into meaningful chunks, ignoring empty lines
def preprocess_text(text):
    raw_chunks = text.strip().split("\n")
    cleaned_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return cleaned_chunks

cleaned_chunks = preprocess_text(knowledge_base)

if not cleaned_chunks:
    print("Warning: Knowledge base is empty!")
    cleaned_chunks = ["No knowledge base information provided."]


# 3. Create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
    chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return chunk_embeddings

chunk_embeddings = create_embeddings(cleaned_chunks)


# 4. Search top matching chunks
def get_top_chunks(query, chunk_embeddings, text_chunks):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)

    top_indices = torch.topk(similarities, k=min(3, len(text_chunks))).indices
    top_chunks = [text_chunks[i] for i in top_indices]
    return top_chunks


client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")


# 5. Handle user prompt and history cleanly
def respond(message, history):
    rag_info = get_top_chunks(message, chunk_embeddings, cleaned_chunks)

    system_message = (
        "You are a helpful assistant that answers questions about hair, skin, and nails.\n\n"
        "Use the following information to provide accurate, friendly, and easy-to-understand responses:\n\n"
        f"{rag_info}\n\n"
        "Your priority is helping the user solve their problem or answer their question. Keep responses clear, "
        "concise, and supportive. If the answer cannot be found in the provided information, let the user know rather than guessing."
    )

    messages = [{"role": "system", "content": system_message}]

    # Safely convert Gradio's history into proper dictionary format
    if history:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, bot_msg = item
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if bot_msg:
                    messages.append({"role": "assistant", "content": bot_msg})
            elif isinstance(item, dict):
                messages.append(item)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()


# 6. Set theme and launch UI
custom_theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="pink",
    neutral_hue="rose",
    radius_size="lg"
).set(
    button_primary_background_fill="#EC4899",
    button_primary_background_fill_hover="#DB2777",
    button_primary_text_color="#FFFFFF",
    block_background_fill="#FDF2F8",
    block_border_width="2px",
    block_border_color="#FBCFE8",
    body_background_fill="#FFF1F2",
)

full_screen_css = """
body, html, .gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}
.gradio-container {
    min-height: 100vh !important;
}
/* Make every Gradio block use the full width */
.contain,
.block,
.gr-block,
.gradio-container > div {
    max-width: 100% !important;
    width: 100% !important;
}
"""

with gr.Blocks(theme=custom_theme, css=full_screen_css) as chatbot:

    gr.Image("pink logo.png", show_label=False)

    gr.ChatInterface(fn=respond)

chatbot.launch()
