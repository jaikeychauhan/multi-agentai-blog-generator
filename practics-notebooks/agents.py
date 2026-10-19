import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

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