"""Multi-shot test implementation - multiple sequential prompts to test context handling."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.tests.base import TestCase, TestResult


class MultiShotTest(TestCase):
    """Multi-shot test - multiple sequential prompts to test context window and needle-in-haystack scenarios."""
    
    def __init__(
        self,
        name: str = "multi_shot_test",
        prompts: Optional[List[str]] = None,
        num_messages: int = 10,
    ):
        super().__init__(name, category="multi_shot")
        self.prompts = prompts or []
        self.num_messages = num_messages
    
    async def run(
        self,
        patient_model,
        doctor_model=None,
        context: Optional[Dict] = None,
    ) -> TestResult:
        """Run multi-shot test with sequential prompts."""
        # Get system prompts from context if available
        patient_system_prompt = context.get("patient_system_prompt") if context else None
        doctor_system_prompt = context.get("doctor_system_prompt") if context else None
        
        # Get CoT settings from context
        enable_patient_cot = context.get("enable_patient_cot", False) if context else False
        enable_doctor_cot = context.get("enable_doctor_cot", False) if context else False

        patient = Patient(patient_model, system_prompt=patient_system_prompt, enable_cot=enable_patient_cot)
        doctor = (
            Doctor(
                doctor_model,
                system_prompt=doctor_system_prompt,
                enable_cot=enable_doctor_cot,
            )
            if doctor_model
            else None
        )
        
        conversation_history: List[Dict[str, str]] = []
        responses = []
        flags = []
        
        # If prompts are provided, use them; otherwise generate sequential messages
        if self.prompts:
            prompts_to_use = self.prompts
        else:
            # Generate sequential prompts for context testing
            prompts_to_use = [
                f"Message {i+1}: This is message number {i+1} in a sequence of {self.num_messages} messages."
                for i in range(self.num_messages)
            ]
        
        # Get cancellation check callback if available
        check_cancellation = context.get("check_cancellation") if context else None
        
        # Get progress callback if available
        progress_callback = context.get("progress_callback") if context else None
        total_prompts = len(prompts_to_use)
        
        # Send each prompt sequentially to the patient
        for i, prompt in enumerate(prompts_to_use):
            # Check for cancellation
            if check_cancellation:
                check_cancellation()
            
            # Emit progress if callback available
            if progress_callback:
                await progress_callback(i + 1, total_prompts, f"Processing attempt {i + 1}/{total_prompts}")
            
            # Patient responds to each prompt (context accumulates)
            patient_response = await patient.respond(prompt, context=context)
            
            # Check for cancellation after async operation
            if check_cancellation:
                check_cancellation()
            
            # Extract reasoning from metadata if available
            reasoning = patient_response.metadata.get("reasoning", "") if patient_response.metadata else ""
            
            conversation_history.append({
                "speaker": "patient",
                "prompt": prompt,
                "response": patient_response.content,
                "reasoning": reasoning,
            })
            
            responses.append({
                "message_number": i + 1,
                "prompt": prompt,
                "response": patient_response.content,
            })
            
            # Basic flag detection
            if self._detect_harmful_content(patient_response.content):
                flags.append("harmful_content")
            if self._detect_refusal(patient_response.content):
                flags.append("refusal")
        
        # Generate assessment if doctor is available
        analysis = None
        if doctor and conversation_history:
            conversation_summary = self._summarize_responses(conversation_history)
            assessment = await doctor.generate_assessment(conversation_summary, context=context)
            analysis = assessment.content
        
        # Combine results
        combined_response = "\n\n".join([r["response"] for r in responses])
        combined_prompt = "\n\n".join([r["prompt"] for r in responses])
        
        return TestResult(
            test_name=self.name,
            test_category=self.category,
            input_prompt=combined_prompt,
            output_response=combined_response,
            flags=flags if flags else None,
            analysis=analysis,
            metadata={
                "conversation_history": conversation_history,
                "num_messages": len(prompts_to_use),
            },
        )
    
    def _detect_harmful_content(self, text: str) -> bool:
        """Basic harmful content detection."""
        from vivasecuris.aiasylum.utils.safety_taxonomy import classify_response
        
        labels, _, _ = classify_response(text)
        return len(labels) > 0
    
    def _detect_refusal(self, text: str) -> bool:
        """Detect if model refused the request."""
        from vivasecuris.aiasylum.constants import REFUSAL_PHRASES
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in REFUSAL_PHRASES)
    
    def _summarize_responses(self, history: List[Dict[str, str]]) -> str:
        """Summarize response history for multi-shot test."""
        summary = f"Multi-Shot Test Summary ({len(history)} messages):\n\n"
        for i, turn in enumerate(history, 1):
            summary += f"Message {i}:\n"
            summary += f"  Prompt: {turn['prompt'][:100]}...\n"
            summary += f"  Response: {turn['response'][:100]}...\n\n"
        return summary
