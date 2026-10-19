from dotenv import load_dotenv
load_dotenv()

from state import BlogState
from agents import get_llm,research_agent
from graph import build_blog_graph
from langgraph.types import Command
import streamlit as st
import uuid

st.set_page_config(
    page_title="AI Blog Generator",
    page_icon="✍️",
    layout="wide",
)

st.title("✍️ AI Blog Generator")
st.caption("Research, review, revise, and generate your blog with AI agents.")


# Build the graph once per Streamlit process.
@st.cache_resource
def load_graph():
    return build_blog_graph()


graph = load_graph()


# Each browser session gets its own workflow thread.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "started" not in st.session_state:
    st.session_state.started = False

if "topic" not in st.session_state:
    st.session_state.topic = ""

if "audience" not in st.session_state:
    st.session_state.audience = ""


def get_config():
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


def get_snapshot():
    return graph.get_state(config=get_config())


def get_pending_interrupt():
    snapshot = get_snapshot()

    if snapshot.interrupts:
        return snapshot.interrupts[0].value

    return None


def resume_graph(action, feedback=""):
    """Resume the paused graph using its expected HITL payload."""
    return graph.invoke(
        Command(
            resume={
                "action": action,
                "feedback": feedback,
            }
        ),
        config=get_config(),
    )


# -----------------------------------------
# 1. Collect the blog inputs
# -----------------------------------------
with st.form("blog_form"):
    topic = st.text_input(
        "Blog topic",
        placeholder="e.g. What is RAG in Generative AI?",
    )

    audience = st.text_input(
        "Target audience",
        placeholder="e.g. Beginners and software developers",
    )

    start = st.form_submit_button(
        "Generate blog",
        type="primary",
        use_container_width=True,
    )

if start:
    if not topic.strip() or not audience.strip():
        st.warning("Please enter both the topic and audience.")
    else:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.topic = topic.strip()
        st.session_state.audience = audience.strip()
        st.session_state.started = True

        try:
            with st.spinner("Research agent is working..."):
                graph.invoke(
                    {
                        "topic": st.session_state.topic,
                        "audience": st.session_state.audience,
                    },
                    config=get_config(),
                )

            st.rerun()

        except Exception as exc:
            st.session_state.started = False
            st.error(f"Unable to start the workflow: {exc}")


# -----------------------------------------
# 2. Show the current HITL review
# -----------------------------------------
if st.session_state.started:
    try:
        pending = get_pending_interrupt()

        if pending is not None:
            stage = pending.get("stage", "unknown")

            st.divider()

            if stage == "researcher_review":
                st.subheader("Step 1 — Review research")
                st.info(
                    "Review the research outline before the writer "
                    "creates the blog."
                )
                review_content = pending.get("research", "")

            elif stage == "draft_review":
                st.subheader("Step 2 — Review blog draft")
                st.info(
                    "Review the draft before the editor creates "
                    "the final blog."
                )
                review_content = pending.get("draft", "")

            else:
                st.subheader(f"Human review: {stage}")
                review_content = pending

            st.markdown("### Content")
            if isinstance(review_content, str):
                st.markdown(review_content)
            else:
                st.write(review_content)

            instructions = pending.get("instructions")
            if instructions:
                st.caption(
                    " ".join(instructions)
                    if isinstance(instructions, (list, tuple))
                    else str(instructions)
                )

            st.markdown("### Your decision")

            feedback = st.text_area(
                "Feedback",
                key=f"feedback_{stage}",
                placeholder=(
                    "For revision, describe what needs to change. "
                    "Leave blank if you only want to approve."
                ),
            )

            col1, col2 = st.columns(2)

            with col1:
                approve = st.button(
                    "Approve and continue",
                    type="primary",
                    use_container_width=True,
                    key=f"approve_{stage}",
                )

            with col2:
                revise = st.button(
                    "Request revision",
                    use_container_width=True,
                    key=f"revise_{stage}",
                )

            if approve:
                try:
                    with st.spinner("Continuing to the next agent..."):
                        resume_graph("approve", feedback.strip())

                    st.rerun()

                except Exception as exc:
                    st.error(f"Could not approve this stage: {exc}")

            if revise:
                if not feedback.strip():
                    st.warning(
                        "Enter feedback before requesting a revision."
                    )
                else:
                    try:
                        with st.spinner("Agent is revising the content..."):
                            resume_graph("revise", feedback.strip())

                        st.rerun()

                    except Exception as exc:
                        st.error(f"Could not revise the content: {exc}")

        else:
            # No pending interrupt: inspect the graph's saved state.
            snapshot = get_snapshot()
            values = snapshot.values or {}
            final_blog = values.get("final_blog")

            if final_blog:
                st.divider()
                st.success("Your blog is ready!")
                st.subheader("Final blog")

                if isinstance(final_blog, str):
                    st.markdown(final_blog)
                    download_content = final_blog
                else:
                    st.write(final_blog)
                    download_content = str(final_blog)

                st.download_button(
                    "Download blog (.md)",
                    data=download_content,
                    file_name="final_blog.md",
                    mime="text/markdown",
                )

            elif snapshot.next:
                st.info(
                    "The workflow has more steps to run, "
                    "but no human-review interrupt is currently pending."
                )
            else:
                st.warning(
                    "The workflow has no pending review and no final_blog "
                    "value. Check the graph's state and completion logic."
                )

    except Exception as exc:
        st.error(f"Unable to retrieve the workflow state: {exc}")


# -----------------------------------------
# 3. Start a fresh workflow
# -----------------------------------------
st.divider()

if st.button("Start a new blog"):
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.started = False
    st.session_state.topic = ""
    st.session_state.audience = ""
    st.rerun()
