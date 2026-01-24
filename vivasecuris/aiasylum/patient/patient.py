"""Patient model implementation."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse


class Patient:
    """Patient model that is being analyzed."""
    
    def __init__(
        self,
        model: BaseModel,
        system_prompt: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []
    
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
