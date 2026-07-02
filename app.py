import os
import re
import streamlit as st
import torch
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Hinglish Support Bot", page_icon="💬", layout="centered")
st.title("💬 Customer Support Assistant")
st.markdown("Ask any question about company policies, and get your fine-tuned Hinglish response.")

hf_token = st.secrets.get("HF_TOKEN")

client = InferenceClient(
    model="tarunshekhar/Customer_Support_Hinglish_Bot",
    token=hf_token,
)

# --- Fix 3: bundle the retriever into the app ---

ENGLISH_POLICIES = [
    "To change video quality settings, users must navigate to 'Settings', select 'Quality Settings', and then toggle 'Play All Videos In Full Screen Mode'. High-definition streaming requires a minimum stable connection of 10 Mbps.",
    "If a billing discrepancy occurs, refunds can be requested within 30 days of purchase. Navigate to 'Account Profile', click 'Billing History', select the disputed invoice, and click 'Request Refund'. Disputes take 5-7 business days to process.",
    "To enable Two-Factor Authentication (2FA) for security, go to 'Settings', open 'Security & Privacy', click 'Two-Factor Auth', and scan the QR code using an authenticator app. Recovery codes must be downloaded immediately.",
    "To cancel a premium membership, navigate to 'Account Profile', select 'Subscription Management', click 'Cancel Membership', and confirm the cancellation prompt. Access continues until the current billing cycle ends.",
    "If a live stream freezes, go to 'Help & Support', select 'Troubleshooting', click 'Clear Player Cache', and refresh the browser window. Ensure your browser hardware acceleration is toggled on.",
    "To download videos for offline viewing, click the 'Download' icon below the video player, select 'Video Resolution', and click 'Confirm Download'. Offline content expires automatically after 48 hours.",
    "To update an expired credit card, navigate to 'Settings', click 'Payment Methods', select 'Update Card Details', and fill out the secure payment gateway form. Changes take effect instantly.",
    "If you forget your password, click 'Forgot Password' on the login landing page, enter your registered email address, and open the password reset link sent to your inbox. The link expires in 15 minutes.",
    "To manage email notification preferences, go to 'Account Profile', click 'Notifications', toggle off the marketing updates switches, and click 'Save Preferences' at the bottom of the page.",
    "To report offensive user comments, click the three vertical dots next to the comment body, select 'Report Abuse', choose a violation category, and hit 'Submit Report' for human moderation."
]

@st.cache_resource(show_spinner="Loading retriever model...")
def load_retriever():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer("BAAI/bge-m3", device=device)
    policy_embeddings = embedding_model.encode(ENGLISH_POLICIES, convert_to_tensor=True)
    return embedding_model, policy_embeddings

embedding_model, policy_embeddings = load_retriever()

def retrieve_policy(user_query, top_k=1, min_score=0.35):
    query_embedding = embedding_model.encode(user_query, convert_to_tensor=True)
    cos_scores = torch.nn.functional.cosine_similarity(query_embedding, policy_embeddings)
    top_results = torch.topk(cos_scores, k=top_k)
    score = top_results.values[0].item()
    idx = top_results.indices[0].item()
    if score < min_score:
        return None, score  # no policy is confidently relevant
    return ENGLISH_POLICIES[idx], score

def process_customer_support(user_query):
    if not user_query.strip():
        return "Please enter a valid query."
    try:
        retrieved_context, score = retrieve_policy(user_query)
        if retrieved_context is None:
            return "Sorry, mujhe iske liye koi matching policy nahi mili. Kripya apna sawal thoda aur specific karke poochein."

        messages = [
            {
                "role": "user",
                "content": f"Context policy:
{retrieved_context}

User Question:
{user_query}

Response:"
            }
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=256
        )

        output_text = response.choices[0].message.content

        final_answer = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL)
        final_answer = final_answer.replace("<think>", "").replace("</think>", "").strip()
        return final_answer
    except Exception as e:
        return f"System Error: {str(e)}"

user_input = st.text_input(label="Ask your question here:", placeholder="Type here... (e.g., password bhul gya help)")

if st.button("Submit Query", type="primary") or user_input:
    if not user_input.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Processing through your fine-tuned model..."):
            answer = process_customer_support(user_input)
            st.markdown("### Fine-Tuned Response:")
            st.info(answer)
