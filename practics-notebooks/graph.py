from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt,Command
from typing import Literal
from state import BlogState
from agents import get_llm,research_agent,writer_agent,editor_agent

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

### Conditional Edges
def route_after_research_review(state:BlogState)->Literal["research","writer"]:
    if state.research_feedback:
        return "research"
    else:
        return "writer"


def route_after_draft_review(state:BlogState)->Literal["edit","writer"]:
    if state.research_feedback and state.revision_count<MAX_REVISION_HISTORY:
        return "writer"
    else:
        return "edit"

### Build and Compile the Graph
def build_blog_graph():
    builder=StateGraph(BlogState)

    ### ADD NODES
    builder.add_node("research",research_node)
    builder.add_node("review_research",human_review_research_node)
    builder.add_node("writer",writer_node)
    builder.add_node("review_draft",human_review_draft_node)
    builder.add_node("edit",editor_node)
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
    builder.add_edge("edit",END)
    graph=builder.compile(checkpointer=InMemorySaver())
    return graph

