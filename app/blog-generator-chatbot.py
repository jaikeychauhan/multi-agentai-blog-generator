from dotenv import load_dotenv
load_dotenv()

from state import BlogState
from agents import get_llm,research_agent
from graph import build_blog_graph
from langgraph.types import Command
import streamlit as stlt
import uuid
# -----------------------------
# Streamlit page configuration
# -----------------------------
stlt.set_page_config(
    page_title="AI Blog Generator",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

stlt.title("✍️AI Blog Generator")
llm=get_llm()

# Building the graph
@stlt.cache_resource
def load_graph():
    return build_blog_graph()


#Each Browser gets its own workflow thread
if "thread_id" not in stlt.session_state:
    stlt.session_state.thread_id=str(uuid.uuid4())

if "started" not in stlt.session_state:
    stlt.session_state.started=False

if "topic" not in stlt.session_state:
    stlt.session_state.topic=""

if "audience" not in stlt.session_state:
    stlt.session_state.audience=""

graph=load_graph()

def getConfiguration():
    return {
        "configurable":{
            "thread_id":stlt.session_state.thread_id
        }
    }

def get_snapshot():
    return graph.get_state(config=getConfiguration())

def get_pending_interrupts():
    snapshot=get_snapshot()
    if snapshot.interrupts:
        return snapshot.interrupts[0].value

    return None

def resume_graph(action:str,feedback:str=""):
    """Resume the paused graph using its expected HITL payload"""
    return graph.invoke(
        Command(resume={
            "action":action,
            "feedback": feedback
        }),
        config=getConfiguration()
    )

### Calling the generate button if input entered
with stlt.form("Blog Form",clear_on_submit=False):
    stlt.markdown("<div class='form-label'>Topic</div>", unsafe_allow_html=True)
    topic = stlt.text_input(
        label="Blog Topic",
        label_visibility="collapsed",
        placeholder="e.g. What is LLM in AI?",
    )
    stlt.markdown("<div class='form-label'>Target Audience</div>", unsafe_allow_html=True)
    audience = stlt.text_input(
        label="Target Audience",
        label_visibility="collapsed",
        placeholder="e.g. Beginners or software developers",
    )
    stlt.markdown("") # Spacing
    
    startGenerateBlogButton = stlt.form_submit_button(
        "✨Generate Blog",
        type="primary",
        use_container_width=True,
    )

#    topic = stlt.text_input("Topic",placeholder="e.g. What is LLM in AI?")
 #   audience = stlt.text_input("Target audience",placeholder="e.g. Beginners or software developers")
    # startGenerateBlogButton = stlt.form_submit_button(
    #         "Generate Blog",
    #         type="primary",
    #         use_container_width=True,
    #     )
if startGenerateBlogButton:
    if not topic.strip() or not audience.strip():
        stlt.warning("Please enter both topic and audience!")
    else:
        stlt.session_state.thread_id = str(uuid.uuid4())
        stlt.session_state.topic = topic.strip()
        stlt.session_state.audience = audience.strip()
        stlt.session_state.started = True
        
        try:
            input_data = {
                "topic": stlt.session_state.topic,
                "audience": stlt.session_state.audience
            }         
            with stlt.spinner("Research Agent is working"):      
                state = graph.invoke(input_data, config=getConfiguration())
            stlt.rerun()
        except Exception as exp:
            stlt.error(f"Unable to start the workflow: {exp}")

### Getting the current HITL response
if stlt.session_state.started:
    try:
        pending_interrupts=get_pending_interrupts()
        stlt.divider()
        if pending_interrupts is not None:
            stage=pending_interrupts.get("stage")
            if stage == "researcher_review":
                stlt.subheader("Step 1- Research Review")
                stlt.info(
                    "Review the Research Outline before the Writer "
                    "Creates the blog."
                )
                review_content=pending_interrupts.get("research","")
            elif stage == "draft_review":
                stlt.subheader("Step 2- Draft Blog Review")
                stlt.info(
                    "Review the Draft blog before the Editor creates the final blog."
                )
                review_content=pending_interrupts.get("draft","")
            else:
                stlt.subheader(f"Human review: {stage}")
                review_content = pending_interrupts

            stlt.markdown("### ---------Content------------")
            if isinstance(review_content,str):
                stlt.markdown(review_content)
            else:
                stlt.write(review_content)

            reviewer_instructions=pending_interrupts.get("instructions")
            if reviewer_instructions:
                if isinstance(reviewer_instructions,(list,tuple)):
                    captionText=" ".join(reviewer_instructions)
                else:
                    captionText=str(reviewer_instructions)

                stlt.caption(captionText)

            stlt.markdown("### Your decision")
            
            feedback = stlt.text_area(
                "Feedback",
                key=f"feedback_{stage}",
                placeholder=(
                    "For revision, describe what needs to change. "
                    "Leave blank if you only want to approve."
                ),
            )
            col1, col2 = stlt.columns(2)
            with col1:
                approve = stlt.button("Approve and continue",type="primary", use_container_width=True, key=f"approve_{stage}",)

            with col2:
                revise = stlt.button("Request revision", use_container_width=True,key=f"revise_{stage}",)


            if approve:
                try:
                    with stlt.spinner("Continuing to the next level."):
                        resume_graph("approve",feedback.strip())
                    stlt.rerun()
                except Exception as exc:
                    stlt.error(f"Could not approve this stage:{exc}")

            if revise:
                if not feedback.strip():
                    stlt.warning("Enter the feedback before requesting a revision.")
                else:
                    try:
                        with stlt.spinner("Agent is revising the content again."):
                            resume_graph("revise",feedback.strip())
                        stlt.rerun()
                    except Exception as exc:
                        stlt.error(f"Could not revise the content: {exc}")
        else:
            #No pending interrupts:
            snapshot=get_snapshot()
            values=snapshot.values or {}
            final_blog=values.get("final_blog")
            if final_blog:
                stlt.divider()
                stlt.success("Your Blog is ready")
                stlt.subheader("Final Blog")

                if isinstance(final_blog,str):
                    stlt.markdown(final_blog)
                else:
                    stlt.write(final_blog)
            else:
                stlt.warning(
                    "The workflow has no pending review and no final blog"
                    )
    except Exception as exc:
        stlt.error(f"Unable to retrieve workflow state: {exc}")

stlt.divider()

if stlt.session_state.started == True:
    if stlt.button("Start New Blog", type="secondary"):
        stlt.session_state.thread_id = str(uuid.uuid4())
        stlt.session_state.started = False
        stlt.session_state.topic = ""
        stlt.session_state.audience = ""
        stlt.rerun()