import os
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# 1. DEFINE APPLICATION STATE
class AgentState(TypedDict):
    messages: List[BaseMessage]
    claim_data: dict
    fraud_score: float
    risk_assessment: str
    status: str  # "Approved", "Human_Review_Needed", "Rejected"

# 2. DEFINE AGENTS & NODES
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def triage_agent(state: AgentState):
    """Parses claim and validates policy."""
    claim_text = state['messages'][-1].content
    # Simulated extraction: in production, use structured output tools
    prompt = f"Extract claim details: amount, policy_id, and incident from: {claim_text}"
    response = llm.invoke([SystemMessage(content="You are a Triage Agent."), HumanMessage(content=prompt)])
    
    # Mock data for demonstration
    return {
        "claim_data": {"amount": 4500, "policy_id": "POL-123", "type": "Auto"},
        "messages": [response]
    }

def fraud_detection_agent(state: AgentState):
    """Checks for historical fraud patterns."""
    policy_id = state['claim_data']['policy_id']
    # Mock check: Check if policy_id has 3+ claims in 6 months
    fraud_score = 0.15 # Low fraud risk
    response = HumanMessage(content=f"Fraud Check Complete. Score: {fraud_score}")
    return {"fraud_score": fraud_score, "messages": [response]}

def risk_assessment_agent(state: AgentState):
    """Evaluates risk and financial impact."""
    amount = state['claim_data']['amount']
    if amount > 10000:
        assessment = "High Severity"
    else:
        assessment = "Standard Severity"
    response = HumanMessage(content=f"Risk Assessment: {assessment}")
    return {"risk_assessment": assessment, "messages": [response]}

def decision_router(state: AgentState):
    """Determines next steps based on risk and fraud."""
    amount = state['claim_data']['amount']
    fraud = state['fraud_score']
    
    if fraud > 0.7:
        return "rejected"
    if amount > 5000 or fraud > 0.4:
        return "human_review"
    return "auto_approve"

# 3. BUILD THE GRAPH
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("triage", triage_agent)
workflow.add_node("fraud_check", fraud_detection_agent)
workflow.add_node("risk_assess", risk_assessment_agent)

# Define Flow
workflow.set_entry_point("triage")
workflow.add_edge("triage", "fraud_check")
workflow.add_edge("fraud_check", "risk_assess")

# Add Conditional Decision Path
workflow.add_conditional_edges(
    "risk_assess",
    decision_router,
    {
        "rejected": END,
        "human_review": END, # In prod, this routes to a human UI/Inbox
        "auto_approve": END
    }
)

# 4. COMPILE AND RUN
app = workflow.compile()

# Example Execution
initial_input = {
    "messages": [HumanMessage(content="My car was hit while parked. Policy POL-123. Estimated repair: $4500.")],
    "claim_data": {},
    "fraud_score": 0.0,
    "risk_assessment": "",
    "status": "pending"
}

for output in app.stream(initial_input):
    for key, value in output.items():
        print(f"--- Node: {key} ---")
        if 'messages' in value:
            print(value['messages'][-1].content)
