from pydantic import BaseModel

class BlogState(BaseModel):
    ### User's Input
    topic: str=""
    audience:str="general readers"

    ### Researcher Output
    research:str=""
    research_feedback:str=""

    ### Writer Output
    draft:str=""
    draft_feedback:str=""

    ### Editor Output
    final_blog:str=""

    ### Metadat
    revision_count:int =0