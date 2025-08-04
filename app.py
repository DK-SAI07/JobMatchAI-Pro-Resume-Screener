# Imports/Core Libraries
import streamlit as st
import docx2txt
import pdfplumber
import re
import google.generativeai as genai
import google.api_core.exceptions
import json
import requests
from streamlit_lottie import st_lottie
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import numpy as np
import json
with open("skill_synonyms.json", "r", encoding="utf-8") as f:
    SKILL_SYNONYMS = json.load(f)

# Load environment variables
from dotenv import load_dotenv
import os

# Load .env file from script directory (works regardless of working directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)

# Also try loading from current directory as fallback
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


# Pre-load NLP Models
@st.cache_resource
def load_nlp_models():
    """Loads the spaCy and the domain-specific Sentence Transformer models."""
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        st.error("spaCy model 'en_core_web_md' not found. Please run 'python -m spacy download en_core_web_md'")
        st.stop()
    # Using the specified domain-specific model
    sentence_model = SentenceTransformer("AventIQ-AI/bert-talentmatchai")
    return nlp, sentence_model

nlp, sentence_model = load_nlp_models()

# Helper Functions
def load_css(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")
 
css_path= pathlib.Path("assets/styles.css")
load_css(css_path)

def load_lottiefile(filepath: str):
    """Loads a Lottie animation from a local JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Animation file not found at: {filepath}")
        st.info("Please make sure the Lottie JSON file is in the same folder as your Python script.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from the Lottie file: {filepath}")
        return None

def extract_text(file):
    """Extracts text from PDF, DOCX, or TXT file."""
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    else:
        return extract_text_from_txt(file)

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def extract_text_from_docx(file):
    try:
        return docx2txt.process(file)
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
    return ""

def extract_text_from_txt(file):
    try:
        return file.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
    return ""

def preprocess_text(text):
    """Deep cleans text for local NLP analysis (Version A)."""
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def minimal_clean(text):
    """Performs light cleaning on text for Gemini (Version B)."""
    return re.sub(r'\s+', ' ', text).strip()

def safe_generate(model, prompt):
    try:
        return model.generate_content(prompt)
    except google.api_core.exceptions.ResourceExhausted:
        st.error("🚫 Gemini API quota exhausted. Please try again later.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {e}")
        st.stop()

def extract_ner_entities(text):
    """Extracts structured entities using spaCy NER."""
    doc = nlp(text)
    entities = {
        "SKILL": [],
        "JOB_TITLE": [],
        "ORG": [],
        "EDU": []
    }
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "PRODUCT", "TECH"]:
            entities["SKILL"].append(ent.text)
        elif ent.label_ == "TITLE":
            entities["JOB_TITLE"].append(ent.text)
        elif ent.label_ == "ORG":
            entities["ORG"].append(ent.text)
        elif ent.label_ in ["DEGREE", "UNIVERSITY"]:
            entities["EDU"].append(ent.text)
    return {k: list(set(v)) for k, v in entities.items()}

def calculate_fuzzy_score(jd_skills, resume_skills):
    """Calculates a fuzzy matching score for skills."""
    if not jd_skills:
        return 0
    total_score = 0
    for skill in jd_skills:
        # Find the best match for each JD skill in the resume's skills
        match = process.extractOne(skill, resume_skills, scorer=fuzz.token_set_ratio)
        if match:
            total_score += match[1] # match[1] is the score (0-100)
    return int(total_score / len(jd_skills))

def normalize_skills(skills, synonym_dict):
    normalized = set()
    for skill in skills:
        skill_lower = skill.lower().strip()
        for canonical, synonyms in synonym_dict.items():
            if skill_lower == canonical or skill_lower in synonyms:
                normalized.add(canonical)
                break
        else:
            normalized.add(skill_lower)  # keep as-is if no match found
    return list(normalized)

def analyze_with_gemini(api_key, model_name, resume_text, jd_text):
    """Analyzes resume against job description using the selected Gemini API model."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        You are an expert HR analyst and a highly skilled resume screening AI. Your task is to provide a detailed, structured analysis of the provided resume against the job description.
        Respond only in a structured JSON format with the following keys:

        1. "match_percentage": An integer from 0 to 100 indicating how well the resume matches the job description.
        2. "profile_overview": A concise, recruiter-style summary (2–6 sentences) that reflects the candidate's key experience from their resume and justifies the match score. For example: "Front-end developer with 5+ years’ experience in scalable UIs. While proficient in JS frameworks, no mention of medical device experience required for this role."
        3. "missing_keywords": A list of crucial skills, technologies, or qualifications mentioned in the job description but not clearly reflected in the resume. Be flexible — do not restrict to a fixed count.
        4. "match_category": A textual category corresponding to the match score. Use only one of the following:
           - "Excellent Fit ✅"
           - "Good Fit ⚠️"
           - "Moderate Fit"
           - "Poor Fit ❌"
        5. "recommendation": A short recommendation that starts with one of these exact labels:
           - "Suitable for shortlisting"
           - "Consider if JD is flexible"
           - "Not recommended for this role"
           After the label, add a – and add a short reasoning of upto 2-4 sentences (e.g., "Suitable for shortlisting – the candidate aligns strongly with required skills but lacks domain-specific certifications.")

        Job Description:
        ---
        {jd_text}
        ---
        Resume:
        ---
        {resume_text}
        ---
        Provide only the JSON object as your response.
        """
        response = safe_generate(model, prompt)
        # response = model.generate_content(prompt)
        json_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_response_text)
    except Exception as e:
        st.error(f"An error occurred with the Gemini API: {e}")
        return None
    
# UI Themes
THEMES = {"dark": """<style> body { font-family: ui-sans-serif, system-ui, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"; } .stApp { background-color: #0d1117; color: #c9d1d9; } h1, h2, h3, h4, h5, h6 { color: #58a6ff; } </style>"""}

# App Initialization
st.set_page_config(page_title="JobMatchAI Pro", page_icon="🤓", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'splash'
if 'results' not in st.session_state: st.session_state.results = []
if 'gemini_api_key' not in st.session_state: st.session_state.gemini_api_key = gemini_api_key or ''
if 'final_jd_raw' not in st.session_state: st.session_state.final_jd_raw = ''
if 'selected_model' not in st.session_state: st.session_state.selected_model = 'gemini-1.5-flash'
if 'on_demand_results' not in st.session_state: st.session_state.on_demand_results = {}

# App Pages
def show_splash_screen():
    st.markdown(THEMES['dark'], unsafe_allow_html=True) 
    lottie_json = load_lottiefile("Animation - 1752042367104.json")
    col1, col2 = st.columns((2, 1.5))
    with col1:
        st.title("Hello there! 👋🏻")
        st.header("Welcome to JobMatchAI Pro!")
        st.subheader("Intelligent Resume Screening, Reimagined.")
        st.markdown("""<div class="splash-description" style="font-size: 21px; line-height: 1.6;"><p><strong>Your intelligent resume screening assistant, built for modern hiring.</p><p>This AI-powered platform is exclusively for <strong>recruiters</strong>, <strong>hiring managers</strong>, and <strong>job seekers</strong> alike.</p><p>Whether you're evaluating candidates or fine-tuning your own resume to stand out, JobMatchAI Pro blends smart NLP with Google Gemini to deliver fast, accurate, and insightful job-to-resume matching — all in just a few clicks.</p></div>""", unsafe_allow_html=True)
        if st.button("Start Resume Analysis", key="start_button"):  
            st.session_state.page = 'main'
            st.rerun()
    with col2:
        if lottie_json:
            st_lottie(lottie_json, height=450, key="resume_animation", loop=True)
        else:
            st.warning("Animation could not be loaded.")

def show_main_app():
    st.markdown(THEMES['dark'], unsafe_allow_html=True)
    st.title("👩🏻‍💻 JobMatchAI Pro: Resume Screening Tool")
    st.write("Drop in resumes and a job description — get instant, AI-powered candidate rankings with deep skill matching.")

    with st.sidebar:
        st.header("Settings & Configuration")
        st.markdown("🔐 Authentication Setup")
        st.markdown("""
        <style>
        .stAlert > div{
            font-size: 9px !important;
            margin-top: -20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.info("Make sure your Gemini API key is set in a .env file like: GEMINI_API_KEY=your-api-key-here")
        st.markdown(
             "<p style='font-size: 14px; color: g   ray;'>"
             "🔒" "  " "Your API key is never stored on our servers or logs. It is only used in memory for this session."
             "</p>",
             unsafe_allow_html=True
             )
        st.session_state.selected_model = st.selectbox("Select Analysis Model", ('gemini-1.5-flash', 'gemini-1.5-pro'), index=0 if st.session_state.selected_model == 'gemini-1.5-flash' else 1)
        st.markdown("[Get your free API key from Google AI Studio](https://aistudio.google.com/app/apikey)")

    col1, col2 = st.columns((1, 2))
    with col1:
        st.header("Step 1: Upload Candidate Resumes")
        uploaded_resumes = st.file_uploader("Upload one or more resumes in PDF, DOCX, or TXT format.", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if uploaded_resumes:
            st.info(f"{len(uploaded_resumes)} resume(s) ready for analysis.")
    with col2:
        st.header("Step 2: Add the Job Description")
        jd_text_input = st.text_area("Paste the complete Job Description here, or Upload a file below.", height=220)
        uploaded_jd = st.file_uploader("Upload a Job Description File", type=["pdf", "docx", "txt"], key="jd_uploader")

    if st.button("Generate Candidate Analysis", key="analyze_button"):
        st.write("") 
        if not st.session_state.gemini_api_key:
            st.warning("No Gemini API key found. Set GEMINI_API_KEY in your .env file to continue.")
        elif not uploaded_resumes:
            st.warning("Please Upload at least One Resume.")
        else:
            final_jd_raw = jd_text_input
            if uploaded_jd:
                final_jd_raw = extract_text(uploaded_jd)
            st.session_state.final_jd_raw = final_jd_raw
            st.session_state.on_demand_results = {}
            
            if not final_jd_raw:
                st.warning("Please provide a Job Description by Pasting or Uploading a File.")
            else:
                with st.spinner("📄 Performing multi-component local NLP pre-screening..."):
                    jd_processed = preprocess_text(final_jd_raw)
                    jd_embedding = sentence_model.encode(jd_processed)
                    jd_entities = extract_ner_entities(final_jd_raw)
                    jd_skills = jd_entities["SKILL"]
                    total_jd_entities = sum(len(v) for v in jd_entities.values())

                    local_results = []
                    for resume_file in uploaded_resumes:
                        resume_raw_text = extract_text(resume_file)
                        if resume_raw_text:
                            resume_processed = preprocess_text(resume_raw_text)
                            resume_embedding = sentence_model.encode(resume_processed)
                            
                            # Semantic Score
                            semantic_score = int(util.pytorch_cos_sim(jd_embedding, resume_embedding).item() * 100)
                            
                            # TF-IDF Score
                            vectorizer = TfidfVectorizer(stop_words='english')
                            tfidf_matrix = vectorizer.fit_transform([jd_processed, resume_processed])
                            tfidf_score = int(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100)
                            
                            # NER Score
                            resume_entities = extract_ner_entities(resume_raw_text)
                            overlap_count = sum(len(set(jd_entities[k]) & set(resume_entities[k])) for k in jd_entities)
                            ner_score = int((overlap_count / total_jd_entities) * 100) if total_jd_entities > 0 else 0
                            
                            # Fuzzy Score
                            resume_skills = normalize_skills(resume_entities["SKILL"], SKILL_SYNONYMS)
                            jd_skills_normalized = normalize_skills(jd_skills, SKILL_SYNONYMS)
                            fuzzy_score = calculate_fuzzy_score(jd_skills_normalized, resume_skills)

                            
                            # Blended Score
                            blended_score = int((semantic_score * 0.5) + (tfidf_score * 0.2) + (ner_score * 0.2) + (fuzzy_score * 0.1))
                            
                            local_results.append({
                                "filename": resume_file.name,
                                "raw_text": minimal_clean(resume_raw_text),
                                "semantic_score": semantic_score,
                                "tfidf_score": tfidf_score,
                                "ner_score": ner_score,
                                "fuzzy_score": fuzzy_score,
                                "blended_score": blended_score,
                                "is_fully_analyzed": False
                            })
                
                with st.spinner("⏳ Hang tight — JobMatchAI is analyzing resumes and generating insights based on the job description."):
                    all_scores = [res["blended_score"] for res in local_results]
                    mean_score = np.mean(all_scores) if all_scores else 0
                    std_score = np.std(all_scores) if all_scores else 0
                    dynamic_threshold = max(25, mean_score - 0.5 * std_score)
                    
                    high_potential_resumes = [res for res in local_results if res["blended_score"] >= dynamic_threshold]
                    
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_resume = {
                            executor.submit(analyze_with_gemini, st.session_state.gemini_api_key, st.session_state.selected_model, res["raw_text"], st.session_state.final_jd_raw): res
                            for res in high_potential_resumes
                        }
                        for future in as_completed(future_to_resume):
                            res_data = future_to_resume[future]
                            try:
                                gemini_result = future.result()
                                if gemini_result:
                                    res_data.update(gemini_result)
                                    res_data["is_fully_analyzed"] = True
                            except Exception as exc:
                                st.error(f'{res_data["filename"]} generated an exception: {exc}')
                    
                    st.session_state.results = local_results
                
                # Display Results or Error
                if local_results:
                    display_results()
                else:
                    st.error("Could not process any resumes. Check file formats, Content, and your API key.")

    # Display Results (only if results exist from previous analysis)
    elif st.session_state.results:
        display_results()

def display_results():
    st.write("") 
    st.success("✅ Analysis Complete! Here’s what we found.")
    
    high_potential_results = sorted(
        [res for res in st.session_state.results if res["is_fully_analyzed"]] + list(st.session_state.on_demand_results.values()),
        key=lambda x: x.get("match_percentage", 0),
        reverse=True
    )
    low_match_results = sorted(
        [res for res in st.session_state.results if not res["is_fully_analyzed"]],
        key=lambda x: x.get("blended_score", 0),
        reverse=True
    )

    st.header("🏅 Top Candidate Insights & Ranking")
    if high_potential_results:
        # Use a set to avoid duplicating on-demand results that are already in the high-potential list
        displayed_filenames = set()
        rank = 1
        for result in high_potential_results:
            if result['filename'] not in displayed_filenames:
                display_full_analysis_card(result, rank)
                displayed_filenames.add(result['filename'])
                rank += 1
    else:
        st.info("ℹ️ No resumes reached the screening threshold. Please review the candidates listed below for manual evaluation.")

    if low_match_results:
        with st.expander("Low-Scoring Candidates (Manual Review Recommended)"):
            for result in low_match_results:
                # Still show the low-match card even if it was analyzed on demand
                display_low_match_card(result)
    
    # Job Seeker Section
    st.write("")  # Add small space before navigation info
    st.info("🧭 You can upload more resumes or try a different job description above to continue exploring.")
    
    st.markdown("""
    <style>
    [data-testid="stExpander"] summary p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    </style>
    """, unsafe_allow_html=True)
    
    with st.expander("Are you a Job Seeker or a Fresher?"):
        st.markdown("""
        <p style="font-size: 16.6px; margin-bottom: 10px;">Make your resume stand out with expert insights and tips:</p>
        """, unsafe_allow_html=True)
        st.markdown("""
        - [7 Resume Tips from HR Pros](https://hyresnap.com/blogs/resume-scan-hr)
        - [ATS Resume Format: Best Practices (YouTube)](https://www.youtube.com/watch?v=ZcuziWFfpQY&t=67s)
        - [Reddit: Real Recruiter Advice](https://www.reddit.com/r/jobs/comments/7y8k6p/im_an_exrecruiter_for_some_of_the_top_companies/)
        """)

def display_full_analysis_card(result, rank):
    if result.get('match_percentage', 0) >= 75: card_class = "results-card-strong"
    elif result.get('match_percentage', 0) >= 50: card_class = "results-card-medium"
    else: card_class = "results-card-low"
    
    with st.container():
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        st.subheader(f"#{rank} – {result['filename']}")
        st.markdown(f"**📊 Match Score:** `{result.get('match_percentage', 'N/A')}%` - **{result.get('match_category', 'N/A')}**")
        st.progress(result.get('match_percentage', 0))
        st.markdown(f"**📄 Profile Overview:**")
        st.markdown(f"_{result.get('profile_overview', 'N/A')}_")
        if result.get('missing_keywords'):
            st.markdown("**❌ Missing Keywords:**")
            st.warning(", ".join(result['missing_keywords']))
        st.markdown(f"**🗂️ Recommendation Note:** {result.get('recommendation', 'N/A')}")
        st.markdown('</div>', unsafe_allow_html=True)

def display_low_match_card(result):
    with st.container():
        st.markdown('<div class="results-card-low">', unsafe_allow_html=True)
        st.subheader(f"{result['filename']}")
        st.markdown(f"**Initial Match Score:** `{result['blended_score']}%` (Semantic: {result['semantic_score']}%, TF-IDF: {result['tfidf_score']}%, NER: {result['ner_score']}%, Fuzzy: {result['fuzzy_score']}%)")
        st.info("This resume has a low pre-screening score.")

        if st.button("Run In-Depth Analysis", key=f"analyze_{result['filename']}", type="secondary"):
            with st.spinner(f"Performing comprehensive match analysis for {result['filename']}…"):
                gemini_result = analyze_with_gemini(st.session_state.gemini_api_key, st.session_state.selected_model, result["raw_text"], st.session_state.final_jd_raw)
                if gemini_result:
                    gemini_result['filename'] = result['filename']
                    st.session_state.on_demand_results[result['filename']] = gemini_result
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Page Router
if st.session_state.page == 'splash':
    show_splash_screen()
else:
    show_main_app()