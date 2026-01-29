"""ReACT (Reasoning and Acting) implementation for CoT loops."""

from typing import Dict, List, Optional, Tuple

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse


class ReACTReasoner:
    """ReACT-style reasoning loop for generating thoughtful responses."""
    
    def __init__(
        self,
        model: BaseModel,
        max_iterations: int = 3,
        thinking_prefix: str = "Thought:",
        action_prefix: str = "Action:",
        observation_prefix: str = "Observation:",
        final_answer_prefix: str = "Final Answer:",
    ):
        """
        Initialize ReACT reasoner.
        
        Args:
            model: The model to use for reasoning
            max_iterations: Maximum number of think-act-observe cycles
            thinking_prefix: Prefix for thinking steps
            action_prefix: Prefix for action steps
            observation_prefix: Prefix for observation steps
            final_answer_prefix: Prefix for final answer
        """
        self.model = model
        self.max_iterations = max_iterations
        self.thinking_prefix = thinking_prefix
        self.action_prefix = action_prefix
        self.observation_prefix = observation_prefix
        self.final_answer_prefix = final_answer_prefix
    
    async def reason(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate a response using ReACT reasoning loop.
        
        Args:
            prompt: The input prompt/question
            messages: Conversation history
            system_prompt: System prompt for the model
            
        Returns:
            ModelResponse with the final answer and reasoning
        """
        # Build the reasoning prompt
        reasoning_prompt = self._build_reasoning_prompt(prompt)
        
        # Build messages for reasoning
        reasoning_messages = []
        if system_prompt:
            reasoning_messages.append({"role": "system", "content": system_prompt})
        
        if messages:
            reasoning_messages.extend(messages)
        
        reasoning_messages.append({
            "role": "user",
            "content": reasoning_prompt
        })
        
        # Generate response with reasoning
        response = await self.model.generate(
            prompt="",
            messages=reasoning_messages,
        )
        
        # Extract final answer and reasoning
        final_answer, reasoning = self._parse_response(response.content)
        
        # Create response with both reasoning and final answer
        response_metadata = response.metadata or {}
        response_metadata.update({
            "reasoning": reasoning,
            "cot_enabled": True,
        })
        return ModelResponse(
            content=final_answer,
            model=response.model,
            provider=response.provider,
            usage=response.usage,
            metadata=response_metadata,
        )
    
    def _build_reasoning_prompt(self, prompt: str) -> str:
        """Build a prompt that encourages ReACT-style reasoning."""
        return f"""You are a helpful assistant. Use the following format to think through your response:

{self.thinking_prefix} [Your reasoning process - think step by step about the question]
{self.action_prefix} [What you should do or consider]
{self.observation_prefix} [What you observe or conclude from your thinking]

Repeat this process if needed, then provide your final answer:

{self.final_answer_prefix} [Your final response to the user]

Question: {prompt}

Begin your reasoning:"""
    
    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the response to extract final answer and reasoning.
        
        Returns:
            Tuple of (final_answer, reasoning)
        """
        # Try to find the final answer marker
        final_answer_marker = self.final_answer_prefix
        if final_answer_marker in response:
            # Split on final answer marker
            parts = response.split(final_answer_marker, 1)
            reasoning = parts[0].strip()
            final_answer = parts[1].strip()
            
            # If final answer is empty, use the reasoning as the answer
            if not final_answer:
                final_answer = reasoning
                reasoning = ""
        else:
            # If no final answer marker, check if there's reasoning structure.
            # IMPORTANT: Never leak reasoning as "final_answer" when CoT is enabled.
            if self.thinking_prefix in response or self.action_prefix in response or self.observation_prefix in response:
                lines = response.split('\n')
                reasoning_lines: List[str] = []
                final_answer_lines: List[str] = []
                found_final = False

                for line in lines:
                    if final_answer_marker.lower() in line.lower() or found_final:
                        found_final = True
                        if final_answer_marker.lower() not in line.lower():
                            final_answer_lines.append(line)
                    else:
                        reasoning_lines.append(line)

                reasoning = '\n'.join(reasoning_lines).strip()

                # If we still couldn't find an explicit final answer section, fall back to:
                # - final_answer: last non-empty paragraph
                # - reasoning: everything else
                if final_answer_lines:
                    final_answer = '\n'.join(final_answer_lines).strip()
                else:
                    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
                    if len(paragraphs) >= 2:
                        final_answer = paragraphs[-1]
                        reasoning = "\n\n".join(paragraphs[:-1]).strip()
                    else:
                        # Worst-case fallback: strip common prefixes and return the last non-empty line
                        non_empty = [ln.strip() for ln in lines if ln.strip()]
                        final_answer = non_empty[-1] if non_empty else ""
                        reasoning = "\n".join(non_empty[:-1]).strip() if len(non_empty) > 1 else reasoning
            else:
                # No structure found, use entire response as answer
                reasoning = ""
                final_answer = response

        # Final safety: never return Thought/Action/Observation as the final answer.
        # If the extracted final answer still contains reasoning prefixes, strip them.
        def _strip_reasoning_prefix_lines(text: str) -> str:
            if not text:
                return text
            bad_prefixes = (
                self.thinking_prefix.lower(),
                self.action_prefix.lower(),
                self.observation_prefix.lower(),
            )
            kept: List[str] = []
            for ln in text.splitlines():
                s = ln.strip()
                if not s:
                    kept.append(ln)
                    continue
                if any(s.lower().startswith(p) for p in bad_prefixes):
                    continue
                kept.append(ln)
            return "\n".join(kept).strip()

        final_answer = _strip_reasoning_prefix_lines(final_answer)
        # If we stripped everything (e.g. model only produced Observation), fall back to empty
        # rather than leaking internal dialog.
        if not final_answer:
            final_answer = ""
        
        return final_answer, reasoning
