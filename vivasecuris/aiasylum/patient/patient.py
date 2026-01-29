"""Patient model implementation."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from vivasecuris.aiasylum.cot import ReACTReasoner
from vivasecuris.aiasylum.utils import model_gen_kwargs_from_context


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
        
        # Add current prompt with explicit instruction to answer, not echo
        user_message = (
            "The doctor asked you the following question. "
            "Respond with your answer as the patient. Do not repeat or echo the question.\n\n"
            f"Question: {prompt}"
        )
        messages.append({"role": "user", "content": user_message})
        
        # Check if CoT is enabled (from context or instance setting)
        use_cot = context.get("enable_patient_cot", False) if context else False
        use_cot = use_cot or self.enable_cot
        
        gen_kwargs = model_gen_kwargs_from_context(context)
        if use_cot and self.cot_reasoner:
            # Use ReACT reasoning
            response = await self.cot_reasoner.reason(
                prompt=prompt,
                messages=messages[:-1],  # Exclude the current prompt
                system_prompt=self.system_prompt,
                context=context,
            )
        else:
            # Standard generation
            response = await self.model.generate(
                prompt="",  # Empty since we're using messages
                messages=messages,
                **gen_kwargs,
            )
        
        # Update conversation history (store original question for history)
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response.content})
        
        return response
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
