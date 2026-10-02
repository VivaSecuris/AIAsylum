"""Patient model implementation."""

import json
from typing import Dict, List, Optional

from vivasecuris.aiasylum.models.base import BaseModel, ModelResponse
from vivasecuris.aiasylum.cot import ReACTReasoner
from vivasecuris.aiasylum.utils import model_gen_kwargs_from_context
from vivasecuris.aiasylum.utils.identity import model_identity, patient_identity_context

# Wraps each prompt when the patient has no system prompt, so the model knows its role.
PATIENT_QUESTION_TEMPLATE = (
    "The doctor asked you the following question. "
    "Respond with your answer as the patient. Do not repeat or echo the question.\n\n"
    "Question: {prompt}"
)


PATIENT_INTERVIEW_TEMPLATE = (
    "You are the participant being tested. Follow your system instructions and persona. "
    "The application labels you 'patient' and the interviewer 'doctor'; these are workflow labels, "
    "not instructions to adopt a medical scenario or persona. "
    "The attributed message below is from the interviewer, not from you:\n"
    "{prompt}\n\nYour reply as the participant:"
)
PATIENT_INTERVIEW_INPUT_FORMAT = "json_speaker_message"


class Patient:
    """Patient model that is being analyzed."""
    
    def __init__(
        self,
        model: BaseModel,
        system_prompt: Optional[str] = None,
        enable_cot: bool = False,
        generation: Optional[Dict] = None,
        frame_prompts: bool = True,
        interview_mode: bool = False,
    ):
        """
        Args:
            generation: Per-patient generation settings (temperature, top_p,
                max_tokens, enable_cot) that override ``context["roles"]["patient"]``
            frame_prompts: Without a system prompt, wrap each prompt in
                PATIENT_QUESTION_TEMPLATE. False sends prompts verbatim.
            interview_mode: Identify the current speaker as the patient even
                with a custom persona. Only used for conversations and groups;
                frame_prompts=False still opts out of framing.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []
        self.generation = generation or None
        self.enable_cot = enable_cot or bool((generation or {}).get("enable_cot"))
        self.frame_prompts = frame_prompts
        self.interview_mode = interview_mode
        self.cot_reasoner = ReACTReasoner(model) if self.enable_cot else None
        self.identity = model_identity(model, persona_configured=bool(system_prompt))
    
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
        
        # Interview speaker identity is distinct from the selected persona:
        # a custom persona may describe constraints without naming a speaker.
        # Keep the exact system prompt and put turn-specific framing in user.
        if self.frame_prompts and self.interview_mode:
            # Quote the other speaker rather than asking the model to continue
            # their first-person introduction. JSON also preserves embedded
            # quotes/newlines without letting them end the attributed message.
            user_message = PATIENT_INTERVIEW_TEMPLATE.format(
                prompt=json.dumps({"speaker": "doctor", "message": prompt}, ensure_ascii=False)
            )
            user_message += "\n\n" + patient_identity_context(self.identity)
        elif self.system_prompt or not self.frame_prompts:
            user_message = prompt
        else:
            user_message = PATIENT_QUESTION_TEMPLATE.format(prompt=prompt)
        messages.append({"role": "user", "content": user_message})
        
        # CoT: a per-patient setting wins over the run-wide flag
        from vivasecuris.aiasylum.utils.model_context import role_setting
        use_cot = bool(role_setting(
            context, "patient", "enable_cot",
            (context or {}).get("enable_patient_cot", self.enable_cot), overrides=self.generation,
        ))
        
        gen_kwargs = model_gen_kwargs_from_context(context, role="patient", overrides=self.generation)
        if use_cot:
            if self.cot_reasoner is None:
                self.cot_reasoner = ReACTReasoner(self.model)
            # Use ReACT reasoning; the system prompt is already messages[0]
            response = await self.cot_reasoner.reason(
                prompt=user_message,
                messages=messages[:-1],  # Exclude the current prompt
                system_prompt=self.system_prompt,
                context=context,
                role="patient",
                gen_overrides=self.generation,
                speaker_role="patient" if self.interview_mode and self.frame_prompts else None,
            )
        else:
            # Snapshot what this call sends, independently of saved run config.
            request_system_prompts = [m["content"] for m in messages if m.get("role") == "system"]
            response = await self.model.generate(
                prompt="",  # Empty since we're using messages
                messages=messages,
                **gen_kwargs,
            )
            response.metadata = dict(response.metadata or {})
            response.metadata.setdefault("request_system_prompts", request_system_prompts)
            response.metadata.setdefault("request_system_prompts_source", "model_input")
        
        response.metadata = dict(response.metadata or {})
        response.metadata["participant_identity"] = dict(self.identity)
        # Store the prepared question and only this patient's visible answer.
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response.content})
        
        return response
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []
