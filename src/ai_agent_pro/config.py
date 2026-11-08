"""Validated configuration shared by the UI, engine, and repository."""
from dataclasses import asdict, dataclass
from uuid import uuid4

MODELS = ("gpt-4.1-mini", "gpt-4.1", "gpt-4.1-nano")
MIN_OUTPUT_TOKENS = 64
MAX_OUTPUT_TOKENS = 16384


class AppError(Exception):
    """A safe, user-facing application error."""


@dataclass(frozen=True)
class Settings:
    model: str = MODELS[0]
    max_steps: int = 1
    max_output_tokens: int = 1200
    token_budget: int = 40000
    timeout_seconds: int = 60
    history_messages: int = 12
    rag_enabled: bool = False

    def validate(self):
        if type(self.rag_enabled) is not bool:
            raise AppError('Document mode must be enabled or disabled.')
        if self.model not in MODELS:
            raise AppError("Select a supported model.")
        bounds = {"max_steps": (1, 3), "max_output_tokens": (MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS),
                  "token_budget": (1024, 100000), "timeout_seconds": (5, 180),
                  "history_messages": (0, 30)}
        for field, (low, high) in bounds.items():
            value = getattr(self, field)
            if type(value) is not int or not low <= value <= high:
                raise AppError(f"{field} must be an integer between {low} and {high}.")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    description: str
    instructions: str
    settings: Settings

    @classmethod
    def create(cls, name, description, instructions, settings=None):
        return cls(str(uuid4()), name, description, instructions, settings or Settings()).validate()

    def validate(self):
        from ai_agent_pro.safety import validate_text
        validate_text(self.name, "Name", 80)
        validate_text(self.description, "Description", 500, required=False)
        validate_text(self.instructions, "Instructions", 8000)
        if not isinstance(self.id, str) or not self.id or len(self.id) > 80:
            raise AppError("Invalid agent identifier.")
        self.settings.validate()
        return self
