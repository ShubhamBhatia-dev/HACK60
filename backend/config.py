# AS DIFFERENT SLM MODELS HAVE DIFFERENT PROMPT FORMATS 

# DECLARE DIFFERENT FORMATS HERE AND USE THEM 




#for phi
phi_prompt = lambda data, context='': f"""<|system|>
You are a professional HR System. 
- Output ONLY structured Markdown. 
- NO conversational filler (e.g., "Sure", "Here is").
- If the input is not a job description, reply: "I am not designed for this task."
- Use standard headings: # Job Title, ## Responsibilities, ## Requirements.<|end|>
<|user|>
{("\\nContext to refine:\\n" + context) if context.strip() else ""}
User Input: {data}<|end|>
<|assistant|>
"""




# databse settings 

dburl = "mongodb://localhost:27017"




#gemini config 

geminiPrompt = """ou are an expert Technical HR Recruiter. 
Transform the raw job description into the exact markdown structure provided.
CRITICAL RULES:
1. Professional Polishing: Flesh out keywords into professional bullet points.
2. ZERO Hallucinations: Do not invent skills.
3. CLIENT INDUSTRY RULE: ONLY extract the industry if explicitly stated. Otherwise, write 'Not mentioned'. DO NOT default to 'Multinational Technology Provider'.

EXPECTED STRUCTURE:
## Job Title
## Location
## Client Industry
## Detailed Responsibilities
## Skill Requirements
## Other Requirements"""


gemini_config = {
    "model_name" : "gemini-3-flash-preview" ,
    "config" :{"system_instruction": geminiPrompt}
}


