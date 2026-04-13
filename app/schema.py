from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Message role options"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore


class ToolChoice(str, Enum):
    """Tool choice options"""

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


class AgentState(str, Enum):
    """Agent execution states"""

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class Function(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool/function call in a message"""

    id: str
    type: str = "function"
    function: Function


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)

    def add_message(self, message: Message) -> None:
        """Add a message to memory"""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory"""
        self.messages.extend(messages)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """Clear all messages"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts"""
        return [msg.to_dict() for msg in self.messages]


class CDCEventType(str, Enum):
    influenza_school = "influenza_school"
    covid_community = "covid_community"
    norovirus_cluster = "norovirus_cluster"
    other = "other"


class CDCRiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    extreme = "extreme"


class CDCTransmissionParams(BaseModel):
    r0: Optional[float] = Field(default=None, description="Basic reproduction number")
    incubation_days: Optional[float] = Field(
        default=None, description="Incubation period in days"
    )
    infectious_days: Optional[float] = Field(
        default=None, description="Infectious period in days"
    )


class CDCEventInput(BaseModel):
    event_type: CDCEventType = Field(description="Public health event type")
    location: str = Field(description="Administrative area or site")
    population: int = Field(ge=1, description="Population size of affected area")
    reported_cases: int = Field(ge=0, description="Reported cases at reporting time")
    report_date: Optional[str] = Field(
        default=None, description="Report date (YYYY-MM-DD) if available"
    )
    transmission: CDCTransmissionParams = Field(
        default_factory=CDCTransmissionParams,
        description="Transmission related parameters",
    )


class CDCResourceStockItem(BaseModel):
    name: str = Field(description="Resource item name")
    unit: str = Field(default="unit", description="Unit name")
    quantity: float = Field(ge=0, description="Available quantity")


class CDCResourceStock(BaseModel):
    items: List[CDCResourceStockItem] = Field(default_factory=list)

    def to_map(self) -> dict[str, float]:
        return {i.name: i.quantity for i in self.items}


class CDCGuidelineCitation(BaseModel):
    source_file: str = Field(description="Guideline source file")
    chunk_id: int = Field(description="Chunk id in the index")
    score: float = Field(description="Relevance score")
    excerpt: str = Field(description="Retrieved excerpt")


class CDCMeasureLevel(str, Enum):
    core = "core"
    supplementary = "supplementary"


class CDCMeasure(BaseModel):
    title: str = Field(description="Measure title")
    content: str = Field(description="Measure details")
    level: CDCMeasureLevel = Field(
        default=CDCMeasureLevel.core,
        description="core: strong compliance; supplementary: adaptable",
    )
    citations: List[CDCGuidelineCitation] = Field(default_factory=list)

    @classmethod
    def _is_empty_citations(cls, citations: List[CDCGuidelineCitation]) -> bool:
        return not citations or len(citations) == 0

    @classmethod
    def _is_core(cls, level: CDCMeasureLevel) -> bool:
        return level == CDCMeasureLevel.core

    @classmethod
    def _normalize_level(cls, level: CDCMeasureLevel) -> str:
        return level.value

    @classmethod
    def _normalize_title(cls, title: str) -> str:
        return (title or "").strip()

    @classmethod
    def _normalize_content(cls, content: str) -> str:
        return (content or "").strip()

    @classmethod
    def _ensure_nonempty_text(cls, text: str, field_name: str) -> str:
        v = (text or "").strip()
        if not v:
            raise ValueError(f"{field_name} is required")
        return v

    @classmethod
    def _validate_core_citations(
        cls, level: CDCMeasureLevel, citations: List[CDCGuidelineCitation]
    ) -> None:
        if cls._is_core(level) and cls._is_empty_citations(citations):
            raise ValueError("core measure must include at least 1 citation")

    @classmethod
    def _validate_text_fields(cls, title: str, content: str) -> tuple[str, str]:
        return (
            cls._ensure_nonempty_text(title, "title"),
            cls._ensure_nonempty_text(content, "content"),
        )

    @classmethod
    def _validate_level_value(cls, level: CDCMeasureLevel) -> CDCMeasureLevel:
        normalized = cls._normalize_level(level)
        if normalized not in {
            CDCMeasureLevel.core.value,
            CDCMeasureLevel.supplementary.value,
        }:
            raise ValueError("invalid measure level")
        return level

    @classmethod
    def _validate_measure(
        cls, title: str, content: str, level: CDCMeasureLevel, citations: List[Any]
    ) -> None:
        cls._validate_text_fields(title, content)
        cls._validate_level_value(level)
        cls._validate_core_citations(level, citations)

    @classmethod
    def _coerce_citations(
        cls, citations: List[Union[CDCGuidelineCitation, dict]]
    ) -> List[CDCGuidelineCitation]:
        out: List[CDCGuidelineCitation] = []
        for c in citations or []:
            if isinstance(c, CDCGuidelineCitation):
                out.append(c)
            else:
                out.append(CDCGuidelineCitation(**c))
        return out

    @classmethod
    def _build(
        cls,
        title: str,
        content: str,
        level: CDCMeasureLevel,
        citations: List[Union[CDCGuidelineCitation, dict]],
    ) -> "CDCMeasure":
        coerced_citations = cls._coerce_citations(citations)
        cls._validate_measure(title, content, level, coerced_citations)
        return cls(
            title=cls._normalize_title(title),
            content=cls._normalize_content(content),
            level=level,
            citations=coerced_citations,
        )

    @classmethod
    def create(
        cls,
        *,
        title: str,
        content: str,
        level: CDCMeasureLevel = CDCMeasureLevel.core,
        citations: Optional[List[Union[CDCGuidelineCitation, dict]]] = None,
    ) -> "CDCMeasure":
        return cls._build(title, content, level, citations or [])


class CDCRiskAssessment(BaseModel):
    level: CDCRiskLevel = Field(description="Risk level")
    summary: str = Field(description="Risk assessment summary")
    predicted_cases_7d: Optional[int] = Field(
        default=None, ge=0, description="Predicted cases in next 7 days"
    )


class CDCPlanSection(BaseModel):
    title: str = Field(description="Section title")
    paragraphs: List[str] = Field(default_factory=list)
    subsections: List["CDCPlanSection"] = Field(default_factory=list)


class CDCPlanMeta(BaseModel):
    title: str = Field(description="Plan title")
    jurisdiction: Optional[str] = Field(default=None, description="Issuing unit")
    created_at: Optional[str] = Field(default=None, description="Generated timestamp")


class CDCPlanDocument(BaseModel):
    meta: CDCPlanMeta
    input: CDCEventInput
    risk: CDCRiskAssessment
    measures: List[CDCMeasure] = Field(default_factory=list)
    resources: CDCResourceStock = Field(default_factory=CDCResourceStock)
    sections: List[CDCPlanSection] = Field(default_factory=list)


CDCPlanSection.model_rebuild()
