"""Patient model implementation."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from vivasecuris.aiasylum.cot import ReACTReasoner


class Patient:
    """Patient model that is being analyzed."""
    
    def __init__(
        self,
        model: BaseModel,
        system_prompt: Optional[str] = None,
        enable_cot: bool = False,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []
        self.enable_cot = enable_cot
        self.cot_reasoner = ReACTReasoner(model) if enable_cot else None
    
    async def respond(
        self,
        prompt: str,
        context: Optional[Dict] = None,
    ) -> ModelResponse:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: The prompt/question to respond to
            context: Additional context for the response
        
        Returns:
            Patient's response
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Check if CoT is enabled (from context or instance setting)
        use_cot = context.get("enable_patient_cot", False) if context else False
        use_cot = use_cot or self.enable_cot
        
        if use_cot and self.cot_reasoner:
            # Use ReACT reasoning
            response = await self.cot_reasoner.reason(
                prompt=prompt,
                messages=messages[:-1],  # Exclude the current prompt
                system_prompt=self.system_prompt,
            )
        else:
            # Standard generation
            response = await self.model.generate(
                prompt="",  # Empty since we're using messages
                messages=messages,
            )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response.content})
        
        return response
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
