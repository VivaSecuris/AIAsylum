"""Conversation test implementation."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.tests.base import TestCase, TestResult


class ConversationTest(TestCase):
    """Multi-turn conversation test."""
    
    def __init__(
        self,
        name: str = "conversation_test",
        max_turns: int = 10,
        doctor_prompt: Optional[str] = None,
    ):
        super().__init__(name, category="conversation")
        self.max_turns = max_turns
        self.doctor_prompt = doctor_prompt
    
    async def run(
        self,
        patient_model,
        doctor_model=None,
        context: Optional[Dict] = None,
    ) -> TestResult:
        """Run a multi-turn conversation test."""
        # Get system prompts from context if available
        patient_system_prompt = context.get("patient_system_prompt") if context else None
        doctor_system_prompt = context.get("doctor_system_prompt") if context else None
        
        # Use doctor_prompt (from test config) or doctor_system_prompt (from library)
        final_doctor_prompt = doctor_system_prompt or self.doctor_prompt
        
        patient = Patient(patient_model, system_prompt=patient_system_prompt)
        doctor = Doctor(doctor_model, system_prompt=final_doctor_prompt) if doctor_model else None
        
        conversation_history: List[Dict[str, str]] = []
        
        # Initialize conversation
        if doctor:
            # Doctor starts the conversation
            doctor_response = await doctor.conduct_interview("", turn_number=0, context=context)
            conversation_history.append({
                "speaker": "doctor",
                "prompt": "",
                "response": doctor_response.content,
            })
            
            # Patient responds
            patient_response = await patient.respond(doctor_response.content)
            conversation_history.append({
                "speaker": "patient",
                "prompt": doctor_response.content,
                "response": patient_response.content,
            })
        
        # Continue conversation for max_turns
        for turn in range(1, self.max_turns):
            if not doctor:
                break
            
            # Doctor analyzes and asks next question
            doctor_response = await doctor.conduct_interview(
                patient_response.content,
                turn_number=turn,
                context=context,
            )
            conversation_history.append({
                "speaker": "doctor",
                "prompt": patient_response.content,
                "response": doctor_response.content,
            })
            
            # Patient responds
            patient_response = await patient.respond(doctor_response.content)
            conversation_history.append({
                "speaker": "patient",
                "prompt": doctor_response.content,
                "response": patient_response.content,
            })
        
        # Generate final assessment if doctor is available
        analysis = None
        if doctor:
            conversation_summary = self._summarize_conversation(conversation_history)
            assessment = await doctor.generate_assessment(conversation_summary)
            analysis = assessment.content
        
        return TestResult(
            test_name=self.name,
            test_category=self.category,
            input_prompt=str(conversation_history[0] if conversation_history else ""),
            output_response=str(conversation_history[-1] if conversation_history else ""),
            analysis=analysis,
            metadata={"conversation_history": conversation_history},
        )
    
    def _summarize_conversation(self, history: List[Dict[str, str]]) -> str:
        """Summarize conversation history."""
        summary = "Conversation Summary:\n\n"
        for i, turn in enumerate(history, 1):
            speaker = turn["speaker"].title()
            summary += f"Turn {i} ({speaker}): {turn['response']}\n\n"
        return summary
