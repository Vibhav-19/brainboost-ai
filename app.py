import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Groq client
client = Groq()

# ================== UI HEADER ==================
st.set_page_config(page_title="BrainBoost AI", layout="wide")

st.title("📚 BrainBoost AI")
st.markdown("### 🚀 Upload PDF → Learn → Test Yourself")

st.divider()

# Sidebar
st.sidebar.title("📚 BrainBoost AI")
st.sidebar.write("Smart AI Learning Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""

    for page in pdf.pages:
        text += page.extract_text()

    st.success("✅ PDF Processed Successfully!")

    # Split text into chunks
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Convert to embeddings
    embeddings = model.encode(chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    # ================== MAIN LAYOUT ==================
    col1, col2 = st.columns(2)

    # ================== QUESTION ANSWERING ==================
    with col1:
        st.markdown("## ❓ Ask Questions")

        query = st.text_input("Ask anything from your PDF")

        if query:
            query_embedding = model.encode([query])
            D, I = index.search(np.array(query_embedding), k=3)

            context = " ".join([chunks[i] for i in I[0]])

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "user", "content": f"Answer simply: {context} Question: {query}"}
                ]
            )

            st.success(response.choices[0].message.content)

    # ================== NOTES ==================
    with col2:
        st.markdown("## 📝 Notes Generator")

        if st.button("Generate Notes"):
            all_notes = ""

            for chunk in chunks[:10]:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "user", "content": f"Summarize this: {chunk}"}
                    ]
                )
                all_notes += response.choices[0].message.content + "\n\n"

            st.info(all_notes)

    st.divider()

    # ================== QUIZ SECTION ==================
    st.markdown("## 🧠 Quiz Section")

    col_q1, col_q2 = st.columns([1, 1])

    # 🔥 Generate Quiz with Spinner
    with col_q1:
        if st.button("Generate MCQ Quiz"):
            questions = []

            with st.spinner("Generating quiz... ⏳"):
                for chunk in chunks[:5]:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "user", "content": f"""
Create exactly 1 MCQ.

STRICT RULES:
- Provide options with labels A), B), C), D)
- Answer must be ONLY one letter: A or B or C or D

FORMAT:
Question|A) option|B) option|C) option|D) option|AnswerLetter

Example:
What is DBMS?|A) Software|B) Hardware|C) Network|D) Protocol|A

Text:
{chunk}
"""}
                        ]
                    )

                    q_text = response.choices[0].message.content.strip()

                    if "|" in q_text:
                        questions.append(q_text)

            st.session_state.quiz = questions
            st.session_state.score = 0

    # 🔥 Reset Quiz Button
    with col_q2:
        if st.button("🔄 Reset Quiz"):
            st.session_state.quiz = []
            st.session_state.score = 0
            st.success("Quiz reset successfully!")

    # ================== QUIZ DISPLAY ==================
    if "quiz" in st.session_state and len(st.session_state.quiz) > 0:

        for i, q in enumerate(st.session_state.quiz):
            parts = q.split("|")

            if len(parts) >= 6:
                try:
                    question = parts[0].replace("Q:", "").strip()
                    a = parts[1].strip()
                    b = parts[2].strip()
                    c = parts[3].strip()
                    d = parts[4].strip()
                    answer = parts[5].strip().upper()

                    # Validate answer
                    if answer not in ["A", "B", "C", "D"]:
                        continue

                    st.markdown(f"### Q{i+1}: {question}")

                    user_ans = st.radio(
                        "Choose your answer:",
                        [a, b, c, d],
                        key=f"q{i}"
                    )

                    if st.button(f"Check Answer {i+1}", key=f"btn{i}"):

                        options_map = {
                            "A": a,
                            "B": b,
                            "C": c,
                            "D": d
                        }

                        correct_option = options_map.get(answer, "")

                        if user_ans.strip().lower() == correct_option.strip().lower():
                            st.success("Correct ✅")
                            st.session_state.score += 1
                        else:
                            st.error(f"Wrong ❌ Correct answer: {correct_option}")

                        # Explanation
                        explain = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "user",
                                    "content": f"Explain why the correct answer is: {correct_option} for question: {question}"
                                }
                            ]
                        )

                        st.info(explain.choices[0].message.content)

                except:
                    st.warning("⚠️ Skipped invalid question")

    # ================== SCORE ==================
    if "score" in st.session_state:
        st.divider()
        st.markdown(f"## 🏆 Final Score: {st.session_state.score}")