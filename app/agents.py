import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import PlagiarismReportSchema
import json
from langchain_core.messages import HumanMessage, SystemMessage

### Get LLM
def get_llm(modelname:str="openai/gpt-oss-20b",temprature:float=0.5):
    api_key=os.getenv("GROQ_API_KEY")
    llm=ChatGroq(model=modelname,temperature=temprature,api_key=api_key)
    return llm

### ********************** Researcher Agent *************************
RESEARCHER_PROMPT=ChatPromptTemplate.from_messages([
    {"role":"system","content":"""
        "You are a researcher agent: Given a blog topic and target audience, produce a clear,"
        "structured research outline. Include:\n"
        "1. 5-7 Key points the blog should cover\n"
        "2. Important facts, stats, or examples for each points\n"
        "3. Suggested angle or hook\n"
        "Be concise. Use bullet points. Do NOT write write full blog yet." 
    """},
    {"role":"user","content":"Topic: {topic}, Audience:{audience},{revision_hints}, Write the research outline now"}
])
def research_agent(llm:ChatGroq,topic:str,audience:str,feedback:str="")->str:
    revision_hints=f"The human provided this feedback for your previous research - please address it: {feedback}"
    if not feedback:
        revision_hints="This is your first attempt."

    chain=RESEARCHER_PROMPT|llm
    result=chain.invoke({
        "topic":topic,
        "audience":audience,
        "revision_hints":revision_hints
    })
    return result.content

### ***********************Writer Agent*******************************
WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {"role":"system","content":"""
        "You are a Blog Writer Agent. Using the research provided, write a complete, engaging blog post.\n"
        "Rules:\n"
        "1. Length: 500-800 words\n"
        "2. Structure: Catchy title, Intro hook, 3-5 sections with h2 headings, conclusion\n"
        "3. Tone: clear, friendly, suited to the target audience\n"
        "4. Use markdown formatting\n"
        "Do NOT add a 'word count' line at the end." 
    """},
    {"role":"user","content":"""
        Topic: {topic},
        Audience:{audience},
        Research Notes:{research}
        {revision_hints}
        Write the full blog post now
        """
     }
])
def writer_agent(llm:ChatGroq,topic:str,audience:str,research:str="",feedback:str="")->str:
    revision_hints=f"The human provided this feedback on your previous draft and asked for these changes: {feedback}. Please apply these changes during writing the blog"
    if not feedback:
        revision_hints="This is your first attempt."

    chain=WRITER_PROMPT | llm
    result=chain.invoke({
        "topic":topic,
        "audience":audience,
        "research":research,
        "revision_hints":revision_hints
    })
    return result.content

### ***********************Editor Agent*******************************
EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    {"role":"system","content":"""
        "You are an EDITOR Agent - The final quality gate before publishing.\n"
        "Take the draft and produce the FINAL polished version. Specifically:\n"        
        "- Fix grammer, spelling, and awkward phrasing\n"
        "- Tighten wordy sentenses\n"
        "- Improve flow and transitions between sections\n"
        "- Make the title and intro more compelling if needed\n"
        "- Keep the same structure and markdown formatting\n"
        "- Blog wording should look like human not AI, and don't use any special chars and company / fancy words.\n"
        "Output only the finaly polished blog post - no commentary." 
    """},
    {"role":"user","content":"""
        Topic: {topic},
        "Draft":{draft}
        Return the polished blog post
        """
     }
])

def editor_agent(llm:ChatGroq,topic:str,draft:str="")->str:
    chain=EDITOR_PROMPT | llm
    result=chain.invoke({
        "topic":topic,
        "draft":draft
    })
    return result.content

### ***********************Duplicate Checker Agent*******************************
system_instruction = """You are a Blog Duplicate Content Checker Agent. Using the content provided by the editor, check the content for duplicacy and plagiarism.
Rules:
1. VERBATIM PATTERNS: Identify any long, highly specific phrasing, clichés, or generic boilerplate text that appears directly copied or unoriginal.
2. CITATION REQUIREMENT: Check all statistics, direct quotes, studies, and non-obvious facts. Every claim must have a clear source attribution or context anchor (e.g., "According to [Source]").
3. PATCHWRITING & PARAPHRASING: Detect shallow rephrasing where the sentence structure matches common online sources without adding original insight.
4. BRAND & TRADEMARK RISKS: Flag any unlicensed use of proprietary terms, copyrighted frameworks, or trademarked material that requires disclaimers.

CRITICAL: Return ONLY valid raw JSON matching this exact schema (no markdown formatting, no conversational text):
{
  "originality_score": <number between 0 and 100>,
  "pass_status": <true if originality_score >= 85, else false>,
  "flagged_issues": [
    {
      "type": "<"Verbatim Phrasing" | "Missing Citation" | "Shallow Paraphrasing" | "Trademark Risk">",
      "text_snippet": "<exact passage from the blog>",
      "reason": "<why this was flagged>",
      "remedy_instruction": "<actionable instruction for the Writer agent to fix it>"
    }
  ],
  "summary_feedback": "<brief overall evaluation for the writer/editor>"
}"""
def duplicate_checker_agent(llm:ChatGroq,topic:str,final_blog:str="")-> dict:
    user_content = f"Topic: {topic}\n\nFinal Blog:\n{final_blog}"
    result = llm.invoke([
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_content)
    ])

    # result=chain.invoke({
        
    #     "topic":topic,
    #     "final_blog":final_blog
    # })
    raw_content = result.content.strip()

    # Clean up markdown code blocks if the LLM wrapped the JSON in ```json ... ```
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    
    raw_content = raw_content.strip()
    try:
        # Validate and parse using Pydantic
        parsed_report = PlagiarismReportSchema.model_validate_json(raw_content)
        return parsed_report.dict()
    except Exception as e:
        print(f"⚠️ JSON Parsing Fallback Error: {e}")
        print(f"Raw response was: {raw_content}")
        # Fallback safe dictionary if parsing fails
        return {
            "originality_score": 50.0,
            "pass_status": False,
            "flagged_issues": [],
            "summary_feedback": "Error parsing JSON report from duplicate checker agent."
        }
    