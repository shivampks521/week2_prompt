
import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from pdf2image import convert_from_path
import pytesseract
import pdfplumber

# Load environment variables
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  #takes api key


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        
        with pdfplumber.open(pdf_path) as pdf:   #try for text base pdf
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Direct text extraction failed: {e}")

    
    print("Falling back to OCR for image-based PDF.")   #try for ocr base image
    try:
        images = convert_from_path(pdf_path)
        for image in images:
            page_text = pytesseract.image_to_string(image)
            text += page_text + "\n"
    except Exception as e:
        print(f"OCR failed: {e}")

    return text.strip()

# Function to get response from Gemini AI
def analyze_resume(resume_text):
    if not resume_text:
        return {"error": "Resume text is required for analysis."}

    model = genai.GenerativeModel("gemini-1.5-flash")

    base_prompt = f"""Act as an experienced HR professional with technical expertise in [specific job role, e.g., Data Science, Data Analyst, Machine Learning Engineer, etc.], your task is to thoroughly review the provided resume.
      Highlight the candidate's key skills, relevant experience, and educational background.
    • Evaluate how well their qualifications align with the requirements of the specific role they are applying for.
    • Based on this analysis, determine if the candidate is a good fit for the role.
After your evaluation, please provide a percentage score that reflects how closely the candidate's profile matches the job role (e.g., "This profile matches the role by 80%")must be in bold and capital letters.
Conclude your assessment with a clear recommendation (e.g., "Recommend for interview," "Does not meet requirements," etc.)must be in bold and capital letter .
    Resume:
    {resume_text}
    """

    response = model.generate_content(base_prompt)

    analysis = response.text.strip()
    return analysis


# Streamlit app

st.set_page_config(page_title="Resume summarizer")
# Title
st.title("Resume summarizer")
st.write("Analyze candidate resume and match it with job descriptions using Google Gemini AI.")

# Only one column for uploading resume
uploaded_file = st.file_uploader("Please Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.success("Resume uploaded successfully!")
else:
    st.warning("Please upload a resume in PDF format.")

# Add a space between sections
st.markdown("<div style= 'padding-top: 10px;'></div>", unsafe_allow_html=True)

if uploaded_file:
    # Save uploaded file locally for processing
    with open("uploaded_resume.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from PDF
    resume_text = extract_text_from_pdf("uploaded_resume.pdf")

    if st.button("Analyze Resume"):
        with st.spinner("Analyzing resume..."):
            try:
                # Analyze resume
                analysis = analyze_resume(resume_text)
                st.success("Analysis complete!")
                st.write(analysis)
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# Footer
st.markdown("---")
st.markdown("""<p style='text-align: center;'>Powered by <b>Streamlit</b> | Developed by <b>Shivam Sharma in colloboration with siddant kadam</b></p>""", unsafe_allow_html=True)
