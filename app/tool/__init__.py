from app.tool.base import BaseTool
from app.tool.cdc_guideline_search import CDCGuidelineSearchTool
from app.tool.cdc_data_api import CDCDataAPITool
from app.tool.cdc_plan_export import CDCPlanExportTool
from app.tool.tool_collection import ToolCollection

try:
    from app.tool.bash import Bash
except Exception:
    Bash = None

try:
    from app.tool.browser_use_tool import BrowserUseTool
except Exception:
    BrowserUseTool = None

try:
    from app.tool.crawl4ai import Crawl4aiTool
except Exception:
    Crawl4aiTool = None

try:
    from app.tool.create_chat_completion import CreateChatCompletion
except Exception:
    CreateChatCompletion = None

try:
    from app.tool.planning import PlanningTool
except Exception:
    PlanningTool = None

try:
    from app.tool.str_replace_editor import StrReplaceEditor
except Exception:
    StrReplaceEditor = None

try:
    from app.tool.terminate import Terminate
except Exception:
    Terminate = None

try:
    from app.tool.web_search import WebSearch
except Exception:
    WebSearch = None


__all__ = [
    "BaseTool",
    "ToolCollection",
    "CDCGuidelineSearchTool",
    "CDCDataAPITool",
    "CDCPlanExportTool",
]

if Bash:
    __all__.append("Bash")
if BrowserUseTool:
    __all__.append("BrowserUseTool")
if Terminate:
    __all__.append("Terminate")
if StrReplaceEditor:
    __all__.append("StrReplaceEditor")
if WebSearch:
    __all__.append("WebSearch")
if CreateChatCompletion:
    __all__.append("CreateChatCompletion")
if PlanningTool:
    __all__.append("PlanningTool")
if Crawl4aiTool:
    __all__.append("Crawl4aiTool")
