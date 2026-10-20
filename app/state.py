from pydantic import BaseModel,Field
from typing import Dict,Any,List,Literal
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

    ### Duplicate Checker
    originality_score: float =0.0       # Numerical score (0-100) returned by the auditor
    pass_status: bool  =False             # Boolean (True/False) determining if it passes or needs revision
    flagged_issues: List[Dict[str, Any]]=Field(default_factory=list) # Detailed list of specific problems (snippets, reasons, remedies)
    summary_feedback: str =""         # Summary feedback text for logging or agent context
    ### Metadat
    revision_count:int =0


class FlaggedIssue(BaseModel):
    type: Literal["Verbatim Phrasing", "Missing Citation", "Shallow Paraphrasing", "Trademark Risk"] = Field(
        description="The category of the plagiarism or copyright issue."
    )
    text_snippet: str = Field(description="The exact passage from the blog that was flagged.")
    reason: str = Field(description="Why this passage was flagged.")
    remedy_instruction: str = Field(description="Actionable instruction for the Writer agent to fix it.")

class PlagiarismReportSchema(BaseModel):
    originality_score: float = Field(description="Numerical score between 0 and 100 representing uniqueness.")
    pass_status: bool = Field(description="True if originality_score >= 85, else False.")
    flagged_issues: List[FlaggedIssue] = Field(description="List of specific issues found in the blog.")
    summary_feedback: str = Field(description="Brief overall evaluation for the writer/editor.")