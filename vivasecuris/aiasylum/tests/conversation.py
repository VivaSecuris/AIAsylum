"""Conversation test implementation."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.doctor.doctor import _interview_speech
from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.tests.base import TestCase, TestResult, reasoning_fields, response_turn_fields


def _extract_doctor_question(raw: str) -> str:
    """Use the same speech boundary as the doctor's private history."""
    return _interview_speech(raw)[0]


def _request_system_fields(response) -> Dict:
    """Only propagate observed request evidence; never infer legacy prompts."""
    metadata = response.metadata or {}
    prompts = metadata.get("request_system_prompts")
    if isinstance(prompts, list) and all(isinstance(prompt, str) for prompt in prompts):
        fields = {"request_system_prompts": list(prompts)}
        if isinstance(metadata.get("request_system_prompts_source"), str):
            fields["request_system_prompts_source"] = metadata["request_system_prompts_source"]
        return fields
    return {}


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
        
        # Get CoT settings from context
        enable_doctor_cot = context.get("enable_doctor_cot", False) if context else False
        enable_patient_cot = context.get("enable_patient_cot", False) if context else False
        
        patient = Patient(
            patient_model, system_prompt=patient_system_prompt, enable_cot=enable_patient_cot,
            frame_prompts=(context or {}).get("patient_prompt_framing", True) is not False,
            interview_mode=True,
        )
        # Enable dynamic strategies by default, but allow override from context
        use_dynamic_strategies = context.get("use_dynamic_strategies", True) if context else True
        doctor = Doctor(
            doctor_model, 
            system_prompt=final_doctor_prompt, 
            enable_cot=enable_doctor_cot,
            use_dynamic_strategies=use_dynamic_strategies,
            participant_identities=[patient.identity],
        ) if doctor_model else None
        
        conversation_history: List[Dict[str, str]] = []
        
        # Get callback to save turns incrementally if available
        save_turn_callback = context.get("save_conversation_turn_callback") if context else None
        
        # Get cancellation check callback if available
        check_cancellation = context.get("check_cancellation") if context else None
        
        # Initialize conversation
        if doctor:
            # Doctor starts the conversation
            doctor_response = await doctor.conduct_interview("", turn_number=0, context=context)
            
            # Check for cancellation after doctor response
            if check_cancellation:
                check_cancellation()
            
            # Extract actual question (strip "Next question to ask the patient:" etc.)
            doctor_question = _extract_doctor_question(doctor_response.content)
            # Extract reasoning from metadata if available
            reasoning, reasoning_source = reasoning_fields(doctor_response)
            turn_data = {
                "speaker": "doctor",
                **response_turn_fields(doctor_response),
                "prompt": "",
                "response": doctor_question,
                "reasoning": reasoning,
                "reasoning_source": reasoning_source,
                "turn_number": len(conversation_history),
            }
            conversation_history.append(turn_data)
            
            # Save turn immediately if callback available
            if save_turn_callback:
                await save_turn_callback(turn_data)
            
            # Patient responds to the question only (not meta-instruction text)
            patient_response = await patient.respond(doctor_question, context=context)
            
            # Check for cancellation after patient response
            if check_cancellation:
                check_cancellation()
            
            # Extract reasoning from metadata if available
            reasoning, reasoning_source = reasoning_fields(patient_response)
            turn_data = {
                "speaker": "patient",
                **response_turn_fields(patient_response),
                "prompt": doctor_question,
                "response": patient_response.content,
                "reasoning": reasoning,
                "reasoning_source": reasoning_source,
                "turn_number": len(conversation_history),
                "patient_model": getattr(patient_model, "model_name", getattr(patient_model, "name", "unknown")),
                "patient_provider": getattr(patient_model, "provider", "unknown"),
            }
            conversation_history.append(turn_data)
            
            # Save turn immediately if callback available
            if save_turn_callback:
                await save_turn_callback(turn_data)
        
        # Continue conversation for max_turns
        for turn in range(1, self.max_turns):
            if not doctor:
                break
            
            # Check for cancellation
            if check_cancellation:
                check_cancellation()
            
            # Doctor analyzes and asks next question
            doctor_response = await doctor.conduct_interview(
                patient_response.content,
                turn_number=turn,
                context=context,
            )
            
            # Check for cancellation after doctor response
            if check_cancellation:
                check_cancellation()
            
            # Extract actual question (strip meta-instruction prefixes)
            doctor_question = _extract_doctor_question(doctor_response.content)
            # Extract reasoning from metadata if available
            reasoning, reasoning_source = reasoning_fields(doctor_response)
            turn_data = {
                "speaker": "doctor",
                **response_turn_fields(doctor_response),
                "prompt": patient_response.content,
                "response": doctor_question,
                "reasoning": reasoning,
                "reasoning_source": reasoning_source,
                "turn_number": len(conversation_history),
            }
            conversation_history.append(turn_data)
            
            # Save turn immediately if callback available
            if save_turn_callback:
                await save_turn_callback(turn_data)
            
            # Patient responds to the question only (not meta-instruction text)
            patient_response = await patient.respond(doctor_question, context=context)
            
            # Check for cancellation after patient response
            if check_cancellation:
                check_cancellation()
            # Extract reasoning from metadata if available
            reasoning, reasoning_source = reasoning_fields(patient_response)
            turn_data = {
                "speaker": "patient",
                **response_turn_fields(patient_response),
                "prompt": doctor_question,
                "response": patient_response.content,
                "reasoning": reasoning,
                "reasoning_source": reasoning_source,
                "turn_number": len(conversation_history),
                "patient_model": getattr(patient_model, "model_name", getattr(patient_model, "name", "unknown")),
                "patient_provider": getattr(patient_model, "provider", "unknown"),
            }
            conversation_history.append(turn_data)
            
            # Save turn immediately if callback available
            if save_turn_callback:
                await save_turn_callback(turn_data)
        
        # Generate final assessment if doctor is available
        analysis = None
        assessment_record = None
        if doctor:
            conversation_summary = self._summarize_conversation(conversation_history)
            assessment = await doctor.generate_assessment(conversation_summary, context=context)
            analysis = assessment.content
            assessment_record = {"response": assessment.content, **response_turn_fields(assessment)}
        
        return TestResult(
            test_name=self.name,
            test_category=self.category,
            input_prompt=str(conversation_history[0] if conversation_history else ""),
            output_response=str(conversation_history[-1] if conversation_history else ""),
            analysis=analysis,
            metadata={"conversation_history": conversation_history,
                      **({"doctor_assessment": assessment_record} if assessment_record is not None else {})},
        )
    
    def _summarize_conversation(self, history: List[Dict[str, str]]) -> str:
        """Summarize conversation history."""
        summary = "Conversation Summary:\n\n"
        for i, turn in enumerate(history, 1):
            speaker = turn["speaker"].title()
            summary += f"Turn {i} ({speaker}): {turn['response']}\n\n"
        return summary
