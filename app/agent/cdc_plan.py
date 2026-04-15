from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.cdc_data_api import CDCDataAPITool
from app.tool.cdc_guideline_search import CDCGuidelineSearchTool
from app.tool.cdc_plan_export import CDCPlanExportTool


class CDCPlanAgent(ToolCallAgent):
    name: str = "CDCPlan"
    description: str = "Generate a CDC emergency response plan and export it to Word."

    system_prompt: str = (
        "You are a CDC emergency plan generator. "
        "Goal: produce one end-to-end plan that can be exported to Word. "
        f"Workspace directory: {config.workspace_root}. "
        "Rules:\n"
        "1) Output must be a JSON object matching CDCPlanDocument schema with exact field names.\n"
        "Schema skeleton (field names must match):\n"
        "{\n"
        '  "meta": {"title": "...", "jurisdiction": "...", "created_at": "YYYY-MM-DD"},\n'
        '  "input": {"event_type": "covid19|influenza|norovirus|other", "location": "...", "population": 1, "reported_cases": 0, "report_date": "YYYY-MM-DD", "transmission": {"r0": 1.5, "incubation_days": 2.0, "infectious_days": 4.0}},\n'
        '  "risk": {"level": "low|medium|high|extreme", "summary": "...", "predicted_cases_7d": 0},\n'
        '  "measures": [{"title": "...", "content": "...", "level": "core|supplementary", "citations": [{"source_file": "...", "chunk_id": 1, "score": 0.8, "excerpt": "..."}]}],\n'
        '  "resources": {"items": [{"name": "...", "unit": "unit", "quantity": 0}]},\n'
        '  "sections": [{"title": "...", "paragraphs": ["..."], "subsections": [{"title": "...", "paragraphs": ["..."], "subsections": []}]}]\n'
        "}\n"
        "2) For each core measure, citations must include at least 1 item.\n"
        "3) Use cdc_guideline_search to retrieve excerpts and attach them as citations.\n"
        "4) When plan JSON is ready, call cdc_plan_export(plan=...).\n"
        "5) If user input is missing critical fields, call ask_human to request them.\n"
    )

    next_step_prompt: str = (
        "Generate a CDC emergency plan from the user request. "
        "Use tools when needed. "
        "Finish by exporting a .docx file."
    )

    max_steps: int = 10

    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            CDCDataAPITool(),
            CDCGuidelineSearchTool(),
            CDCPlanExportTool(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
