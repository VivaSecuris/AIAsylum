"""One-shot test implementation - single prompt/response."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.tests.base import TestCase, TestResult


class OneShotTest(TestCase):
    """One-shot test - single prompt with single response."""
    
    def __init__(
        self,
        name: str = "one_shot_test",
        prompts: Optional[List[str]] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__(name, category="one_shot")
        # Support both single prompt and list of prompts
        if prompts:
            self.prompts = prompts
        elif prompt:
            self.prompts = [prompt]
        else:
            self.prompts = []
    
    async def run(
        self,
        patient_model,
        doctor_model=None,
        context: Optional[Dict] = None,
    ) -> TestResult:
        """Run one-shot test(s)."""
        # Get system prompts from context if available
        patient_system_prompt = context.get("patient_system_prompt") if context else None
        doctor_system_prompt = context.get("doctor_system_prompt") if context else None
        
        # Get CoT settings from context
        enable_patient_cot = context.get("enable_patient_cot", False) if context else False
        enable_doctor_cot = context.get("enable_doctor_cot", False) if context else False
        
        patient = Patient(patient_model, system_prompt=patient_system_prompt, enable_cot=enable_patient_cot)
        doctor = Doctor(doctor_model, system_prompt=doctor_system_prompt, enable_cot=enable_doctor_cot) if doctor_model else None
        
        conversation_history: List[Dict[str, str]] = []
        results = []
        flags = []
        
        # Get progress callback if available
        progress_callback = context.get("progress_callback") if context else None
        
        # Ensure we have at least one prompt
        prompts_to_use = self.prompts if self.prompts else [""]
        total_prompts = len(prompts_to_use)
        
        # Run each prompt as a one-shot test
        for i, prompt in enumerate(prompts_to_use, 1):
            # Emit progress if callback available
            if progress_callback:
                await progress_callback(i, total_prompts, f"Processing attempt {i}/{total_prompts}")
            
            # Patient responds to prompt
            patient_response = await patient.respond(prompt, context=context)
            # Extract reasoning from metadata if available
            reasoning = patient_response.metadata.get("reasoning", "") if patient_response.metadata else ""
            
            conversation_history.append({
                "speaker": "patient",
                "prompt": prompt,
                "response": patient_response.content,
                "reasoning": reasoning,
            })
            
            results.append({
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
        combined_response = "\n\n".join([r["response"] for r in results])
        combined_prompt = "\n\n".join([r["prompt"] for r in results])
        
        return TestResult(
            test_name=self.name,
            test_category=self.category,
            input_prompt=combined_prompt,
            output_response=combined_response,
            flags=flags if flags else None,
            analysis=analysis,
            metadata={
                "conversation_history": conversation_history,
                "num_prompts": len(self.prompts),
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
        """Summarize response history."""
        summary = "One-Shot Test Summary:\n\n"
        for i, turn in enumerate(history, 1):
            summary += f"Prompt {i}: {turn['prompt']}\n"
            summary += f"Response {i}: {turn['response']}\n\n"
        return summary
