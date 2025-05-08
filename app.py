import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
from Courses import recommend_courses
import json

# Load environment variables
load_dotenv()

# Configure Google Gemini AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set the GOOGLE_API_KEY environment variable")
    st.stop()
genai.configure(api_key=api_key)

# Simulated user database (in-memory)
user_data = []

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to calculate ATS score
def calculate_ats_score(resume_text, job_description):
    if not resume_text or not job_description:
        return 0, set()  # Return tuple with empty set for no matches
    
    # Convert texts to lowercase and split into words
    resume_words = set(word.lower() for word in resume_text.split() if len(word) > 2)
    jd_words = set(word.lower() for word in job_description.split() if len(word) > 2)

    # Find matching words
    matched_words = resume_words.intersection(jd_words)
    
    # Calculate score
    score = (len(matched_words) / len(jd_words)) * 100 if jd_words else 0
    
    return round(score, 2), matched_words

# Function to get response from Gemini AI
def analyze_resume(resume_text, job_description=None):
    if not resume_text:
        return {"error": "Resume text is required for analysis."}
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        base_prompt = """
        As an experienced HR professional with technical expertise, analyze the following resume.
        Please provide:
        1. Overall evaluation of the candidate's profile
        2. Key skills assessment
        3. Skill improvement recommendations
        4. Suggested courses for skill enhancement
        5. Strengths and weaknesses analysis
        Resume:
        {resume_text}
        """
        if job_description:
            base_prompt += """
            Job Description Comparison:
            {job_description}
            Additional Analysis:
            - Skills matching with job requirements
            - Experience alignment
            - Qualification fit
            - Areas for improvement specific to this role
            """
        response = model.generate_content(base_prompt.format(
            resume_text=resume_text,
            job_description=job_description
        ))
        return response.text.strip()
    except Exception as e:
        st.error(f"Error in AI analysis: {str(e)}")
        return None

def extract_resume_data(resume_text):
    # Define default structure at the start of the function
    default_data = {
        "personal_info": {
            "name": "",
            "email": "",
            "phone": "",
            "location": ""
        },
        "skills": [],
        "education": [],
        "experience": [],
        "certifications": []
    }

    if not resume_text:
        return default_data

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """
        Extract key information from this resume and return it in a simple JSON format.
        Only include information that is explicitly present in the resume.
        Format your response as a simple JSON object with these exact fields:
        {
            "personal_info": {
                "name": "extracted name",
                "email": "extracted email",
                "phone": "extracted phone",
                "location": "extracted location"
            },
            "skills": ["skill1", "skill2"],
            "education": [{"degree": "degree name", "institution": "school name", "year": "year", "gpa": "gpa"}],
            "experience": [{"title": "job title", "company": "company name", "duration": "duration", "responsibilities": ["resp1", "resp2"]}],
            "certifications": ["cert1", "cert2"]
        }
        
        Resume text: {resume_text}
        """
        
        safety_config = genai.GenerationConfig(
            temperature=0.1,
            top_p=0.8,
            top_k=40
        )
        
        response = model.generate_content(
            prompt.format(resume_text=resume_text),
            generation_config=safety_config
        )
        
        # Get the response text and clean it
        response_text = response.text.strip()
        
        # Clean up the response
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        try:
            parsed_data = json.loads(response_text)
            
            # Ensure all required fields exist
            for key in default_data:
                if key not in parsed_data:
                    parsed_data[key] = default_data[key]
                    
            # Ensure personal_info has all required fields
            for field in default_data["personal_info"]:
                if field not in parsed_data.get("personal_info", {}):
                    parsed_data.setdefault("personal_info", {})[field] = ""
                    
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Response text: {response_text}")
            return default_data
            
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        return default_data

# Update the display section to handle potential None values
def display_resume_data(resume_data):
    if not resume_data:
        st.error("Could not extract resume information")
        return
        
    st.markdown("<h3 style='color:#2980B9;'>📑 Extracted Information</h3>", unsafe_allow_html=True)
    
    
    
    
    # Display skills
    skills = resume_data.get('skills', [])
    if skills:
        st.markdown("<h4 style='color:#27AE60;'>🛠️ Skills</h4>", unsafe_allow_html=True)
        st.write(", ".join(skills))
    
    # Display education
    education = resume_data.get('education', [])
    if education:
        st.markdown("<h4 style='color:#27AE60;'>🎓 Education</h4>", unsafe_allow_html=True)
        for edu in education:
            if any(edu.values()):
                st.markdown(f"""
                    - **{edu.get('degree', 'Degree')}**
                    - Institution: {edu.get('institution', 'N/A')}
                    - Year: {edu.get('year', 'N/A')}
                    - GPA: {edu.get('gpa', 'N/A')}
                """)
    
    # Display experience
    experience = resume_data.get('experience', [])
    if experience:
        st.markdown("<h4 style='color:#27AE60;'>💼 Experience</h4>", unsafe_allow_html=True)
        for exp in experience:
            if any(exp.values()):
                st.markdown(f"""
                    - **{exp.get('title', 'Position')}** at {exp.get('company', 'Company')}
                    - Duration: {exp.get('duration', 'N/A')}
                """)
                if exp.get('responsibilities'):
                    st.markdown("**Key Responsibilities:**")
                    for resp in exp['responsibilities']:
                        st.markdown(f"  - {resp}")
    
    # Display certifications
    certifications = resume_data.get('certifications', [])
    if certifications:
        st.markdown("<h4 style='color:#27AE60;'>📜 Certifications</h4>", unsafe_allow_html=True)
        for cert in certifications:
            st.markdown(f"- {cert}")

def analyze_job_matching(resume_text, job_description):
    try:
        # Extract keywords from job description
        jd_words = set(word.lower() for word in job_description.split() if len(word) > 2)
        resume_words = set(word.lower() for word in resume_text.split() if len(word) > 2)
        
        # Find matches and missing keywords
        matched_keywords = jd_words.intersection(resume_words)
        missing_keywords = jd_words - resume_words
        
        # Analyze matching sections
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
        Analyze the resume against the job description and provide:
        1. Key skills matching analysis
        2. Experience level matching
        3. Required qualifications matching
        4. Specific improvement suggestions
        
        Resume: {resume_text}
        Job Description: {job_description}
        
        Format your response in a clear, structured way with sections and bullet points.
        """
        
        response = model.generate_content(prompt)
        analysis = response.text.strip()
        
        return {
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "analysis": analysis
        }
    except Exception as e:
        print(f"Error in job matching analysis: {str(e)}")
        return None

def calculate_detailed_scores(resume_text, job_description, resume_data):
    try:
        # Extract keywords and skills
        jd_words = set(word.lower() for word in job_description.split() if len(word) > 2)
        resume_words = set(word.lower() for word in resume_text.split() if len(word) > 2)
        resume_skills = set(skill.lower() for skill in resume_data.get('skills', []))
        
        # Calculate different scores
        # 1. Overall ATS Score (30% weight)
        matched_words = jd_words.intersection(resume_words)
        ats_score = (len(matched_words) / len(jd_words)) * 100 if jd_words else 0
        
        # 2. Skills Match Score (40% weight)
        jd_skills = set(word.lower() for word in job_description.split() if len(word) > 2)
        matched_skills = jd_skills.intersection(resume_skills)
        skills_score = (len(matched_skills) / len(jd_skills)) * 100 if jd_skills else 0
        
        # 3. Experience Level Score (30% weight)
        experience = resume_data.get('experience', [])
        experience_score = min(100, len(experience) * 20)  # 20 points per year of experience, capped at 100
        
        # Calculate weighted total score
        total_score = (ats_score * 0.3) + (skills_score * 0.4) + (experience_score * 0.3)
        
        return {
            "total_score": round(total_score, 2),
            "ats_score": round(ats_score, 2),
            "skills_score": round(skills_score, 2),
            "experience_score": round(experience_score, 2),
            "matched_skills": matched_skills,
            "missing_skills": jd_skills - resume_skills
        }
    except Exception as e:
        print(f"Error in detailed scoring: {str(e)}")
        return None

def analyze_soft_skills_and_sentiment(resume_text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = """
        Analyze the following resume text for soft skills and writing tone. Provide:
        
        1. Soft Skills Analysis:
        - List all detected soft skills (e.g., leadership, communication, teamwork)
        - Rate each skill's prominence (low, medium, high)
        
        2. Writing Style Analysis:
        - Formality level (1-10)
        - Professionalism (1-10)
        - Action-orientation (1-10)
        - Clarity (1-10)
        
        3. Overall Tone Assessment:
        - Confidence level
        - Key tone characteristics
        - Areas for improvement
        
        Format the response as a JSON object with these exact fields:
        {
            "soft_skills": [{"skill": "skill name", "level": "prominence level", "evidence": "brief example"}],
            "writing_style": {
                "formality": number,
                "professionalism": number,
                "action_orientation": number,
                "clarity": number
            },
            "tone_assessment": {
                "confidence_level": "description",
                "characteristics": ["characteristic1", "characteristic2"],
                "improvements": ["improvement1", "improvement2"]
            }
        }
        
        Resume text: {text}
        """
        
        response = model.generate_content(prompt.format(text=resume_text))
        
        # Clean and parse the response
        response_text = response.text.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
            
        return json.loads(response_text)
    except Exception as e:
        print(f"Error in soft skills analysis: {str(e)}")
        return None

def analyze_resume_layout(resume_text):
    if not resume_text:
        return None
        
    try:
        model = genai.GenerativeModel("gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        prompt = """
        Analyze the resume's layout and formatting. Provide a detailed assessment of:

        1. Layout Analysis:
        - Section organization
        - Content hierarchy
        - White space usage
        - Section completeness
        
        2. Formatting Consistency:
        - Font usage patterns
        - Bullet point style
        - Date formats
        - Header styles
        
        3. Design Issues:
        - Identify any inconsistencies
        - Highlight missing elements
        - Flag potential readability issues
        
        4. Specific Recommendations:
        - Layout improvements
        - Formatting fixes
        - Design enhancements
        
        Format the response as a JSON object with these exact fields:
        {
            "layout_analysis": {
                "strengths": ["strength1", "strength2"],
                "weaknesses": ["weakness1", "weakness2"]
            },
            "formatting_issues": [
                {"issue": "description", "severity": "high/medium/low", "fix": "solution"}
            ],
            "design_recommendations": [
                {"area": "area name", "suggestion": "improvement suggestion", "impact": "high/medium/low"}
            ]
        }
        """
        
        response = model.generate_content([
            {"text": prompt},
            {"text": f"Resume text:\n{resume_text}"}
        ])
        
        # Clean and parse the response
        response_text = response.text.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        # Parse JSON with error handling
        try:
            result = json.loads(response_text)
            # Ensure all required fields exist
            required_fields = {
                "layout_analysis": {"strengths": [], "weaknesses": []},
                "formatting_issues": [],
                "design_recommendations": []
            }
            for field, default in required_fields.items():
                if field not in result:
                    result[field] = default
                    
            return result
        except json.JSONDecodeError as e:
            st.error(f"Error parsing layout analysis response: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in layout analysis: {str(e)}")
        return None

def generate_candidate_summary(resume_text, job_description=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        prompt = """
        Generate a professional candidate summary based on the resume. The summary should:
        1. Highlight key qualifications
        2. Emphasize relevant skills
        3. Showcase major achievements
        4. Present experience level
        5. Include career focus
        
        Format the response as a JSON object with these exact fields:
        {
            "professional_summary": "A concise 2-3 sentence overview",
            "key_qualifications": ["qualification1", "qualification2"],
            "core_skills": ["skill1", "skill2"],
            "major_achievements": ["achievement1", "achievement2"],
            "experience_level": "description",
            "career_focus": "description"
        }
        
        Keep the summary professional, concise, and impactful.
        """
        
        if job_description:
            prompt += "\nJob Description:\n" + job_description
        
        response = model.generate_content([
            {"text": prompt},
            {"text": f"Resume text:\n{resume_text}"}
        ])
        
        # Clean and parse the response
        response_text = response.text.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        # Parse JSON with error handling
        try:
            result = json.loads(response_text)
            # Ensure all required fields exist
            required_fields = {
                "professional_summary": "",
                "key_qualifications": [],
                "core_skills": [],
                "major_achievements": [],
                "experience_level": "",
                "career_focus": ""
            }
            for field, default in required_fields.items():
                if field not in result:
                    result[field] = default
                    
            return result
        except json.JSONDecodeError as e:
            st.error(f"Error parsing candidate summary: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error generating candidate summary: {str(e)}")
        return None

def analyze_ai_enhancements(resume_text, job_description=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        prompt = """
        Analyze the resume for potential AI/ML enhancements and fine-tuning opportunities. Consider:
        1. Domain-specific terminology and context
        2. Industry-specific skills and requirements
        3. Technical depth and complexity
        4. Potential areas for model improvement
        
        Format the response as a JSON object with these exact fields:
        {
            "domain_analysis": {
                "primary_domain": "main industry/field",
                "technical_depth": "assessment of technical complexity",
                "specialized_terminology": ["term1", "term2"]
            },
            "enhancement_opportunities": [
                {
                    "area": "specific area for improvement",
                    "potential_impact": "high/medium/low",
                    "suggestion": "specific enhancement suggestion"
                }
            ],
            "fine_tuning_recommendations": {
                "dataset_suggestions": ["suggestion1", "suggestion2"],
                "model_improvements": ["improvement1", "improvement2"],
                "context_enhancements": ["enhancement1", "enhancement2"]
            }
        }
        
        Focus on practical, implementable improvements.
        """
        
        if job_description:
            prompt += "\nJob Description:\n" + job_description
        
        response = model.generate_content([
            {"text": prompt},
            {"text": f"Resume text:\n{resume_text}"}
        ])
        
        # Clean and parse the response
        response_text = response.text.strip()
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        # Parse JSON with error handling
        try:
            result = json.loads(response_text)
            # Ensure all required fields exist
            required_fields = {
                "domain_analysis": {
                    "primary_domain": "",
                    "technical_depth": "",
                    "specialized_terminology": []
                },
                "enhancement_opportunities": [],
                "fine_tuning_recommendations": {
                    "dataset_suggestions": [],
                    "model_improvements": [],
                    "context_enhancements": []
                }
            }
            for field, default in required_fields.items():
                if field not in result:
                    result[field] = default
                    
            return result
        except json.JSONDecodeError as e:
            st.error(f"Error parsing AI enhancements analysis: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error generating AI enhancements analysis: {str(e)}")
        return None

# Streamlit app
st.set_page_config(page_title="AI Resume Analyzer", page_icon="��", layout="wide")

st.markdown("""
    <style>
        .animated-bar {
            height: 10px;
            background: linear-gradient(270deg, #00f260, #0575e6);
            background-size: 400% 400%;
            animation: moveBar 2s ease infinite;
            border-radius: 10px;
            margin: 10px 0 20px;
        }
        @keyframes moveBar {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        .stTextArea textarea {
            background-color: #fdfefe;
            color: #2c3e50;
            font-size: 16px;
            border: 2px solid #2980b9;
            border-radius: 10px;
            padding: 12px;
        }
        .stFileUploader > div {
            background-color: #fefefe;
            border: 2px dashed #27ae60;
            border-radius: 10px;
            padding: 12px;
        }
        .stAlert {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .analysis-card {
            background-color: #F8F9FA;
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.8em;
            color: white;
            display: inline-block;
        }
        .badge-high { background-color: #E74C3C; }
        .badge-medium { background-color: #F39C12; }
        .badge-low { background-color: #27AE60; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='display: flex; align-items: center;'>
        <h1 style='margin-right: auto; color: #2C3E50;'>📄 AI Resume Analyzer</h1>
    </div>
    <p style='color: #7F8C8D;'>Analyze your resume and match it with job descriptions using Google Gemini AI.</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area("Enter Job Description:", placeholder="Paste the job description here...")

if uploaded_file is not None:
    st.success("✅ Resume uploaded successfully!")
else:
    st.warning("⚠️ Please upload a resume in PDF format.")

if uploaded_file:
    try:
        temp_file = "uploaded_resume.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        resume_text = extract_text_from_pdf(temp_file)

        if not resume_text.strip():
            st.error("❌ Could not extract text from the PDF. Please ensure the PDF contains readable text.")
            st.stop()

        if st.button("🔍 Analyze Resume", use_container_width=True):
            try:
                st.markdown('<div class="animated-bar"></div>', unsafe_allow_html=True)
                with st.spinner("Analyzing resume..."):
                    # Extract structured data
                    resume_data = extract_resume_data(resume_text)
                    display_resume_data(resume_data)

                    # Continue with existing analysis
                    analysis = analyze_resume(resume_text, job_description)
                    ats_score, matched_keywords = calculate_ats_score(resume_text, job_description)

                    if analysis:
                        st.success("🎯 Analysis complete!")
                        st.markdown("<h3 style='color:#2980B9;'>📊 Analysis Results</h3>", unsafe_allow_html=True)
                        st.markdown(f"""
                            <div style='background-color:#ECF0F1; padding:20px; border-radius:10px;'>
                                <pre style='white-space: pre-wrap; font-family: "Courier New", monospace;'>{analysis}</pre>
                            </div>
                        """, unsafe_allow_html=True)

                        # Display ATS Score
                        st.markdown("<h3 style='color:#2980B9;'>📊 ATS Score</h3>", unsafe_allow_html=True)
                        st.progress(ats_score / 100)
                        st.markdown(f"<h2 style='text-align:center;'>{ats_score}%</h2>", unsafe_allow_html=True)
                        
                        if matched_keywords:
                            st.markdown("<h4 style='color:#27AE60;'>✅ Matched Keywords</h4>", unsafe_allow_html=True)
                            st.write(", ".join(matched_keywords))

                        # Display Candidate Summary
                        st.markdown("<h3 style='color:#3498DB; margin-top:2rem'>👤 Candidate Summary</h3>", unsafe_allow_html=True)
                        
                        # Generate and display candidate summary
                        candidate_summary = generate_candidate_summary(resume_text, job_description)
                        
                        if candidate_summary:
                            # Professional Summary
                            st.markdown("""
                                <div class="analysis-card" style="background-color:#EBF5FB; border-left:4px solid #3498DB;">
                                    <h4 style="color:#2C3E50; margin-bottom:15px;">Professional Overview</h4>
                                    <p style="color:#2C3E50; line-height:1.6;">{}</p>
                                </div>
                            """.format(candidate_summary['professional_summary']), unsafe_allow_html=True)
                            
                            # Key Qualifications and Core Skills
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                    <div class="analysis-card">
                                        <h4 style="color:#2C3E50; margin-bottom:15px;">Key Qualifications</h4>
                                    </div>
                                """, unsafe_allow_html=True)
                                for qual in candidate_summary['key_qualifications']:
                                    st.markdown(f"""
                                        <div style="background-color:#D5F5E3; padding:10px; border-radius:5px; margin:8px 0;">
                                            ✓ {qual}
                                        </div>
                                    """, unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("""
                                    <div class="analysis-card">
                                        <h4 style="color:#2C3E50; margin-bottom:15px;">Core Skills</h4>
                                    </div>
                                """, unsafe_allow_html=True)
                                for skill in candidate_summary['core_skills']:
                                    st.markdown(f"""
                                        <div style="background-color:#D5F5E3; padding:10px; border-radius:5px; margin:8px 0;">
                                            ✓ {skill}
                                        </div>
                                    """, unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)

                            # Major Achievements
                            st.markdown("""
                                <div class="analysis-card">
                                    <h4 style="color:#2C3E50; margin-bottom:15px;">Major Achievements</h4>
                                </div>
                            """, unsafe_allow_html=True)
                            for achievement in candidate_summary['major_achievements']:
                                st.markdown(f"""
                                    <div style="background-color:#FEF9E7; padding:10px; border-radius:5px; margin:8px 0;">
                                        ⭐ {achievement}
                                    </div>
                                """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Experience Level and Career Focus
                            col3, col4 = st.columns(2)
                            
                            with col3:
                                st.markdown("""
                                    <div class="analysis-card">
                                        <h4 style="color:#2C3E50; margin-bottom:15px;">Experience Level</h4>
                                        <p style="color:#2C3E50;">{}</p>
                                    </div>
                                """.format(candidate_summary['experience_level']), unsafe_allow_html=True)
                            
                            with col4:
                                st.markdown("""
                                    <div class="analysis-card">
                                        <h4 style="color:#2C3E50; margin-bottom:15px;">Career Focus</h4>
                                        <p style="color:#2C3E50;">{}</p>
                                    </div>
                                """.format(candidate_summary['career_focus']), unsafe_allow_html=True)

                        # Display AI/ML Enhancements
                        st.markdown("<h3 style='color:#8E44AD; margin-top:2rem'>🤖 AI/ML Enhancements</h3>", unsafe_allow_html=True)
                        
                        # Generate and display AI enhancements analysis
                        ai_enhancements = analyze_ai_enhancements(resume_text, job_description)
                        
                        if ai_enhancements:
                            # Domain Analysis
                            st.markdown("""
                                <div class="analysis-card" style="background-color:#F5EEF8; border-left:4px solid #8E44AD;">
                                    <h4 style="color:#2C3E50; margin-bottom:15px;">Domain Analysis</h4>
                                    <div style="margin-bottom:10px;">
                                        <strong>Primary Domain:</strong> {}
                                    </div>
                                    <div style="margin-bottom:10px;">
                                        <strong>Technical Depth:</strong> {}
                                    </div>
                                    <div>
                                        <strong>Specialized Terminology:</strong>
                                        <div style="margin-top:5px;">
                                            {}
                                        </div>
                                    </div>
                                </div>
                            """.format(
                                ai_enhancements['domain_analysis']['primary_domain'],
                                ai_enhancements['domain_analysis']['technical_depth'],
                                ", ".join(ai_enhancements['domain_analysis']['specialized_terminology'])
                            ), unsafe_allow_html=True)
                            
                            # Enhancement Opportunities
                            st.markdown("""
                                <div class="analysis-card">
                                    <h4 style="color:#2C3E50; margin-bottom:15px;">Enhancement Opportunities</h4>
                            """, unsafe_allow_html=True)
                            for opp in ai_enhancements['enhancement_opportunities']:
                                impact = opp['potential_impact'].lower()
                                st.markdown(f"""
                                    <div style="background-color:#F9EBEA; padding:10px; border-radius:5px; margin:8px 0; border-left:4px solid {'#E74C3C' if impact == 'high' else '#F39C12' if impact == 'medium' else '#27AE60'}">
                                        <div style="display:flex; justify-content:space-between; align-items:center">
                                            <span style="font-weight:500">{opp['area']}</span>
                                            <span class="badge badge-{impact}">IMPACT: {impact.upper()}</span>
                                        </div>
                                        <p style="margin:8px 0 0 0; color:#7F8C8D">💡 {opp['suggestion']}</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Fine-Tuning Recommendations
                            st.markdown("""
                                <div class="analysis-card">
                                    <h4 style="color:#2C3E50; margin-bottom:15px;">Fine-Tuning Recommendations</h4>
                            """, unsafe_allow_html=True)
                            
                            # Dataset Suggestions
                            st.markdown("""
                                <div style="margin-bottom:15px;">
                                    <h5 style="color:#2C3E50;">Dataset Suggestions</h5>
                            """, unsafe_allow_html=True)
                            for suggestion in ai_enhancements['fine_tuning_recommendations']['dataset_suggestions']:
                                st.markdown(f"""
                                    <div style="background-color:#E8F8F5; padding:10px; border-radius:5px; margin:8px 0;">
                                        📊 {suggestion}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Model Improvements
                            st.markdown("""
                                <div style="margin-bottom:15px;">
                                    <h5 style="color:#2C3E50;">Model Improvements</h5>
                            """, unsafe_allow_html=True)
                            for improvement in ai_enhancements['fine_tuning_recommendations']['model_improvements']:
                                st.markdown(f"""
                                    <div style="background-color:#E8F8F5; padding:10px; border-radius:5px; margin:8px 0;">
                                        🔧 {improvement}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Context Enhancements
                            st.markdown("""
                                <div style="margin-bottom:15px;">
                                    <h5 style="color:#2C3E50;">Context Enhancements</h5>
                            """, unsafe_allow_html=True)
                            for enhancement in ai_enhancements['fine_tuning_recommendations']['context_enhancements']:
                                st.markdown(f"""
                                    <div style="background-color:#E8F8F5; padding:10px; border-radius:5px; margin:8px 0;">
                                        🎯 {enhancement}
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Display Visual Resume Analysis
                        st.markdown("<h3 style='color:#9B59B6; margin-top:2rem'>🎨 Visual Resume Analysis</h3>", unsafe_allow_html=True)
                        
                        # Perform layout analysis
                        layout_analysis = analyze_resume_layout(resume_text)
                        
                        if layout_analysis:
                            # Create container for analysis content
                            with st.container():
                                # Display Layout Strengths and Weaknesses in columns
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("""
                                        <div class="analysis-card">
                                            <h4 style="color:#27AE60">✨ Layout Strengths</h4>
                                    """, unsafe_allow_html=True)
                                    
                                    for strength in layout_analysis['layout_analysis']['strengths']:
                                        st.markdown(f"""
                                            <div style="background-color:#D5F5E3; padding:10px; border-radius:5px; margin:8px 0;">
                                                ✓ {strength}
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("""
                                        <div class="analysis-card">
                                            <h4 style="color:#E74C3C">🔍 Areas for Improvement</h4>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    for weakness in layout_analysis['layout_analysis']['weaknesses']:
                                        st.markdown(f"""
                                            <div style="background-color:#FADBD8; padding:10px; border-radius:5px; margin:8px 0;">
                                                ⚠ {weakness}
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)

                                # Add spacing between sections
                                st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

                                # Display Formatting Issues
                                st.markdown("""
                                    <div style='margin-bottom:20px;'>
                                        <h5 style='color:#2C3E50;'>📝 Formatting Issues</h5>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                for issue in layout_analysis['formatting_issues']:
                                    severity = issue['severity'].lower()
                                    st.markdown(f"""
                                        <div class="analysis-card" style="border-left:4px solid {'#E74C3C' if severity == 'high' else '#F39C12' if severity == 'medium' else '#27AE60'}">
                                            <div style="display:flex; justify-content:space-between; align-items:center">
                                                <span style="font-weight:500">{issue['issue']}</span>
                                                <span class="badge badge-{severity}">{severity.upper()}</span>
                                            </div>
                                            <p style="margin:8px 0 0 0; color:#7F8C8D">💡 {issue['fix']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                                # Add spacing between sections
                                st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

                                # Display Design Recommendations
                                st.markdown("<h4 style='margin-top:2rem'>🎯 Design Recommendations</h4>", unsafe_allow_html=True)
                                
                                for rec in layout_analysis['design_recommendations']:
                                    impact = rec['impact'].lower()
                                    st.markdown(f"""
                                        <div class="analysis-card" style="border-left:4px solid {'#27AE60' if impact == 'high' else '#F39C12' if impact == 'medium' else '#7F8C8D'}">
                                            <div style="display:flex; justify-content:space-between; align-items:center">
                                                <span style="font-weight:500">{rec['area']}</span>
                                                <span class="badge badge-{impact}">IMPACT: {impact.upper()}</span>
                                            </div>
                                            <p style="margin:8px 0 0 0; color:#7F8C8D">💡 {rec['suggestion']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                        # Display Recommended Courses
                        st.markdown("<h3 style='color:#16A085; margin-top:2rem'>📚 Recommended Courses</h3>", unsafe_allow_html=True)
                        
                        # Extract skills for course recommendations
                        skills = []
                        if resume_data and 'skills' in resume_data:
                            skills = resume_data['skills']
                        if job_description:
                            # Add skills from job description
                            jd_skills = set(word.lower() for word in job_description.split() if len(word) > 2)
                            skills.extend(list(jd_skills))
                        
                        if skills:
                            courses = recommend_courses(skills)
                            if courses:
                                st.markdown("""
                                    <div class="analysis-card" style="background-color:#E8F6F3; border-left:4px solid #16A085;">
                                        <h4 style="color:#2C3E50; margin-bottom:15px;">Based on your skills and job requirements</h4>
                                """, unsafe_allow_html=True)
                                
                                for title, url in courses:
                                    st.markdown(f"""
                                        <div style="background-color:#D5F5E3; padding:15px; border-radius:5px; margin:10px 0;">
                                            <div style="display:flex; align-items:center;">
                                                <span style="margin-right:10px;">🎓</span>
                                                <div>
                                                    <a href="{url}" target="_blank" style="color:#16A085; text-decoration:none; font-weight:500;">{title}</a>
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.info("No specific course recommendations available at this time.")
                        else:
                            st.info("Please upload a resume or enter a job description to get course recommendations.")

                        st.markdown("<h3 style='color:#27AE60;'>📈 Key Insights</h3>", unsafe_allow_html=True)
                        
                        # Display key insights in a clean format without graphs
                        st.markdown("""
                            <div style='background-color:#F8F9FA; padding:20px; border-radius:10px; margin:10px 0;'>
                                <h4 style='color:#2C3E50; margin-bottom:15px;'>Resume Overview</h4>
                                <ul style='list-style-type:none; padding-left:0;'>
                                    <li style='margin-bottom:10px;'>📊 Technical Skills Match</li>
                                    <li style='margin-bottom:10px;'>💼 Experience Level Assessment</li>
                                    <li style='margin-bottom:10px;'>🎯 Role Alignment</li>
                                    <li style='margin-bottom:10px;'>📈 Growth Potential</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

                        # Add Visual Resume Analysis Section with proper spacing and structure
                        st.markdown("""
                            <div style='margin-top:30px; margin-bottom:20px;'>
                                <h3 style='color:#9B59B6; margin-bottom:20px;'>
                                    <span style='margin-right:10px;'>🎨</span>Visual Resume Analysis
                                </h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Perform layout analysis
                        layout_analysis = analyze_resume_layout(resume_text)
                        
                        if layout_analysis:
                            # Create container for analysis content
                            with st.container():
                                # Display Layout Strengths and Weaknesses in columns
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("""
                                        <div class="analysis-card">
                                            <h4 style="color:#27AE60">✨ Layout Strengths</h4>
                                    """, unsafe_allow_html=True)
                                    
                                    for strength in layout_analysis['layout_analysis']['strengths']:
                                        st.markdown(f"""
                                            <div style="background-color:#D5F5E3; padding:10px; border-radius:5px; margin:8px 0;">
                                                ✓ {strength}
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("""
                                        <div class="analysis-card">
                                            <h4 style="color:#E74C3C">🔍 Areas for Improvement</h4>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    for weakness in layout_analysis['layout_analysis']['weaknesses']:
                                        st.markdown(f"""
                                            <div style="background-color:#FADBD8; padding:10px; border-radius:5px; margin:8px 0;">
                                                ⚠ {weakness}
                                            </div>
                                        """, unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)

                                # Add spacing between sections
                                st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

                                # Display Formatting Issues
                                st.markdown("""
                                    <div style='margin-bottom:20px;'>
                                        <h5 style='color:#2C3E50;'>📝 Formatting Issues</h5>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                for issue in layout_analysis['formatting_issues']:
                                    severity = issue['severity'].lower()
                                    st.markdown(f"""
                                        <div class="analysis-card" style="border-left:4px solid {'#E74C3C' if severity == 'high' else '#F39C12' if severity == 'medium' else '#27AE60'}">
                                            <div style="display:flex; justify-content:space-between; align-items:center">
                                                <span style="font-weight:500">{issue['issue']}</span>
                                                <span class="badge badge-{severity}">{severity.upper()}</span>
                                            </div>
                                            <p style="margin:8px 0 0 0; color:#7F8C8D">💡 {issue['fix']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                                # Add spacing between sections
                                st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

                                # Display Design Recommendations
                                st.markdown("<h4 style='margin-top:2rem'>🎯 Design Recommendations</h4>", unsafe_allow_html=True)
                                
                                for rec in layout_analysis['design_recommendations']:
                                    impact = rec['impact'].lower()
                                    st.markdown(f"""
                                        <div class="analysis-card" style="border-left:4px solid {'#27AE60' if impact == 'high' else '#F39C12' if impact == 'medium' else '#7F8C8D'}">
                                            <div style="display:flex; justify-content:space-between; align-items:center">
                                                <span style="font-weight:500">{rec['area']}</span>
                                                <span class="badge badge-{impact}">IMPACT: {impact.upper()}</span>
                                            </div>
                                            <p style="margin:8px 0 0 0; color:#7F8C8D">💡 {rec['suggestion']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")

        if os.path.exists(temp_file):
            os.remove(temp_file)

    except Exception as e:
        st.error(f"❗ Error processing file: {str(e)}")

st.markdown("---")
st.markdown("""
    <p style='text-align: center;'>Powered by <b>Streamlit</b> and <b>Google Gemini AI</b> </p>
""", unsafe_allow_html=True)
