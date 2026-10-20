from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command
from typing import Literal
from state import BlogState
from agents import get_llm,research_agent,writer_agent,editor_agent,duplicate_checker_agent

MAX_REVISION_HISTORY=3

### Nodes -EDGES
# Defining Nodes
def research_node(state:BlogState):
    """Researcher agent generates (or revises) the research outline"""
    llm=get_llm()
    research_data=research_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        feedback=state.research_feedback
    )
    state.research=research_data
    state.research_feedback=""
    return state

def human_review_research_node(state:BlogState):
    """Pause and ask the human to approve the research or send the feedback"""
    decision=interrupt({
        "stage":"researcher_review",
        "research":state.research,
        "instructions":(
          "Reply with 'approve' to continue to writing"  ,
          "or decribe what to change to send it back to the research"
        )
    })
    if(isinstance(decision,dict)):
        action=decision.get("action","approve")
        feedback=decision.get("feedback","")
    else:
        text=str(decision)
        action="approve" if text.lower() in ["approve","ok","approved","yes","proceed","process","continue",""] else "revise"
        feedback="" if action == "approve" else text

    state.research_feedback=feedback
    return state

def writer_node(state:BlogState):
    """Writer Agent produce the full draft blog post(or revise it)"""
    llm=get_llm()
    draft_data=writer_agent(
        llm=llm,
        topic=state.topic,
        audience=state.audience,
        feedback=state.draft_feedback
    )
    state.draft=draft_data
    state.draft_feedback=""
    return state

def human_review_draft_node(state:BlogState):
    """Pause and ask the human to approve the draft or send the feedback"""
    decision=interrupt({
        "stage":"draft_review",
        "draft":state.draft,
        "instructions":(
          "Reply with 'approve' to continue to send the editor"  ,
          "or decribe what to change to send it back to the writer"
        )
    })
    if(isinstance(decision,dict)):
        action=decision.get("action","approve")
        feedback=decision.get("feedback","")
    else:
        text=str(decision)
        action="approve" if text.lower() in ["approve","ok","approved","yes","proceed","process","continue",""] else "revise"
        feedback="" if action == "approve" else text

    if action == "revise":
        state.revision_count += 1

    state.draft_feedback=feedback
    return state

def editor_node(state:BlogState):
    llm=get_llm()
    final=editor_agent(
        llm=llm,
        topic=state.topic,
        draft=state.draft
    )
    state.final_blog=final
    return state

def duplicate_checker_node(state:BlogState):
    llm=get_llm()
    report=duplicate_checker_agent(
        llm=llm,
        topic=state.topic,
        final_blog=state.final_blog
    )
    return {
        "originality_score": report["originality_score"],
        "pass_status": report["pass_status"],
        "flagged_issues": report["flagged_issues"],
        "plagiarism_report": report["summary_feedback"]
    }

def human_review_plagiarism_node(state: BlogState):
    """Pause and ask the human to review the plagiarism report and decide next steps"""
    decision = interrupt({
        "stage": "plagiarism_review",
        "originality_score": state.originality_score,
        "pass_status": state.pass_status,
        "flagged_issues": state.flagged_issues,
        "summary_feedback":state.summary_feedback,
        "final_blog": state.final_blog,
        "instructions": (
            "Reply with 'approve' to publish the blog despite the warning, "
            "or describe what changes the writer needs to make to fix the flagged issues."
        )
    })    
    if isinstance(decision, dict):
        action = decision.get("action", "approve")
        feedback = decision.get("feedback", "")
    else:
        text = str(decision)
        # If user says approve/ok, we proceed to END, otherwise treat text as feedback to revise
        action = "approve" if text.lower() in ["approve", "ok", "approved", "yes", "proceed", "continue", ""] else "revise"
        feedback = "" if action == "approve" else text

    # If revising, store the feedback so the writer agent knows what to fix
    if action == "revise":
        state.revision_count += 1
        state.draft_feedback = f"Plagiarism Fix Required: {feedback}. Specific issues to address: {state.flagged_issues}"
    
    return state
### Conditional Edges
def route_after_research_review(state:BlogState)->Literal["research","writer"]:
    if state.research_feedback:
        return "research"
    else:
        return "writer"


def route_after_draft_review(state:BlogState)->Literal["edit","writer"]:
    if state.draft_feedback and state.revision_count<MAX_REVISION_HISTORY:
        return "writer"
    else:
        return "edit"

def route_after_plagiarism_review(state: BlogState) -> Literal["writer", "approved"]:
    # If draft_feedback was populated, route back to writer to revise
    if state.draft_feedback:
        return "writer"
    else:
        return "approved"

### Build and Compile the Graph
def build_blog_graph():
    builder=StateGraph(BlogState)

    ### ADD NODES
    builder.add_node("research",research_node)
    builder.add_node("review_research",human_review_research_node)
    builder.add_node("writer",writer_node)
    builder.add_node("review_draft",human_review_draft_node)
    builder.add_node("edit",editor_node)
    builder.add_node("duplicate_checker", duplicate_checker_node)
    builder.add_node("review_plagiarism",human_review_plagiarism_node)
    ### ADD EDGES
    builder.add_edge(START,"research")
    builder.add_edge("research","review_research")
    builder.add_conditional_edges("review_research",
                                route_after_research_review,
                                {"research":"research","writer":"writer"}                                
                                )
    builder.add_edge("writer","review_draft")
    builder.add_conditional_edges("review_draft",
                                route_after_draft_review,
                                {"writer":"writer","edit":"edit"}                                
                                )
    builder.add_edge("edit","duplicate_checker")
    builder.add_edge("duplicate_checker","review_plagiarism")
    builder.add_conditional_edges("review_plagiarism",route_after_plagiarism_review,
                     {
                        "approved": END,
                        "writer": "writer" 
                     }
                )
    graph=builder.compile(checkpointer=InMemorySaver())
    return graph

