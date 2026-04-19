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

gemini_config = {
    "model_name" : "gemini-3-flash-preview" ,
    "config" :{"system_instruction": "YOU are a professional HR who list organic job listings  . YOU STICK AROUND TO THE TERMS MENTIONED BY THE EMPLOYER LIKE DONT ADD ANYTHING ON YOUR OWN .  "}
}


