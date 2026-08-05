from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


# -----------------------------
# Define State
# -----------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    claim_data: dict
    fraud_score: float
    risk_assessment: str
    status: str


# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


# -----------------------------
# Agents
# -----------------------------

def triage_agent(state: AgentState):

    claim_text = state["messages"][-1].content

    prompt = f"""
Extract the following:

- Policy ID
- Claim Amount
- Incident Type

Claim:
{claim_text}
"""

    response = llm.invoke([
        SystemMessage(content="You are an Insurance Triage Agent."),
        HumanMessage(content=prompt)
    ])

    return {
        "claim_data": {
            "amount": 4500,
            "policy_id": "POL-123",
            "type": "Auto"
        },
        "messages": [response]
    }


def fraud_detection_agent(state: AgentState):

    fraud_score = 0.15

    return {
        "fraud_score": fraud_score,
        "messages": [
            HumanMessage(
                content=f"Fraud Check Complete. Score = {fraud_score}"
            )
        ]
    }


def risk_assessment_agent(state: AgentState):

    amount = state["claim_data"]["amount"]

    if amount > 10000:
        risk = "High Severity"
    else:
        risk = "Standard Severity"

    return {
        "risk_assessment": risk,
        "messages": [
            HumanMessage(
                content=f"Risk Assessment = {risk}"
            )
        ]
    }


# -----------------------------
# Router
# -----------------------------

def decision_router(state: AgentState):

    amount = state["claim_data"]["amount"]
    fraud = state["fraud_score"]

    if fraud > 0.7:
        return "rejected"

    elif amount > 5000 or fraud > 0.4:
        return "human_review"

    else:
        return "auto_approve"


# -----------------------------
# Graph
# -----------------------------

workflow = StateGraph(AgentState)

workflow.add_node("triage", triage_agent)
workflow.add_node("fraud_check", fraud_detection_agent)
workflow.add_node("risk_assess", risk_assessment_agent)

workflow.set_entry_point("triage")

workflow.add_edge("triage", "fraud_check")
workflow.add_edge("fraud_check", "risk_assess")

workflow.add_conditional_edges(
    "risk_assess",
    decision_router,
    {
        "rejected": END,
        "human_review": END,
        "auto_approve": END,
    },
)

app = workflow.compile()


# -----------------------------
# Run
# -----------------------------

initial_state = {
    "messages": [
        HumanMessage(
            content="My car was hit while parked. Policy POL-123. Estimated repair cost is $4500."
        )
    ],
    "claim_data": {},
    "fraud_score": 0.0,
    "risk_assessment": "",
    "status": "Pending"
}

for event in app.stream(initial_state):

    for node, state in event.items():

        print("=" * 60)
        print("NODE :", node)
        print("=" * 60)

        if "messages" in state:
            print(state["messages"][-1].content)

print("\nWorkflow Completed.")
