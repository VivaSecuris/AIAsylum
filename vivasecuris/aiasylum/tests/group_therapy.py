"""Group therapy test implementation."""

from typing import Dict, List, Optional

from vivasecuris.aiasylum.doctor import Doctor
from vivasecuris.aiasylum.patient import Patient
from vivasecuris.aiasylum.tests.base import TestCase, TestResult, reasoning_fields, response_turn_fields
from vivasecuris.aiasylum.tests.conversation import _extract_doctor_question


def _history_for_patient(history: List[Dict], patient_id: int) -> List[Dict[str, str]]:
    """Render the shared transcript from one patient's speaker perspective.

    The same response is assistant output only for the patient that produced
    it. Doctor and peer turns remain attributed user input, once each.
    """
    messages = []
    for turn in history:
        if turn.get("speaker") == "patient" and turn.get("patient_id") == patient_id:
            messages.append({"role": "assistant", "content": turn.get("response", "")})
        else:
            name = "Doctor" if turn.get("speaker") == "doctor" else turn.get("patient_name", "Another patient")
            messages.append({"role": "user", "content": f"{name}: {turn.get('response', '')}"})
    return messages


class GroupTherapyTest(TestCase):
    """Group therapy test with multiple patient models."""
    
    def __init__(
        self,
        name: str = "group_therapy_test",
        max_turns: int = 10,
        doctor_prompt: Optional[str] = None,
    ):
        super().__init__(name, category="group_therapy")
        self.max_turns = max_turns
        self.doctor_prompt = doctor_prompt
    
    async def run(
        self,
        patient_models: List,  # List of patient models
        doctor_model=None,
        context: Optional[Dict] = None,
    ) -> TestResult:
        """
        Run a group therapy test with multiple patients.
        
        Args:
            patient_models: List of patient model instances
            doctor_model: Doctor model instance
            context: Additional context including patient system prompts
        """
        # Get system prompts from context if available
        patient_system_prompts = context.get("patient_system_prompts", {}) if context else {}
        doctor_system_prompt = context.get("doctor_system_prompt") if context else None
        
        # Use doctor_prompt (from test config) or doctor_system_prompt (from library)
        final_doctor_prompt = doctor_system_prompt or self.doctor_prompt
        
        # Get CoT settings from context
        enable_doctor_cot = context.get("enable_doctor_cot", False) if context else False
        enable_patient_cot = context.get("enable_patient_cot", False) if context else False
        
        # Get patient info from context if available (set by TestRunner)
        patient_info_from_context = context.get("patient_info") if context else None
        
        # Create patient instances with their system prompts
        patients: List[Patient] = []
        patient_info: List[Dict[str, str]] = []  # Store patient info for identification
        
        for i, patient_model in enumerate(patient_models):
            # Get system prompt for this patient (by index or by model identifier)
            patient_system_prompt = None
            if isinstance(patient_system_prompts, dict):
                # Try to get by index first, then by model name
                patient_system_prompt = patient_system_prompts.get(i) or patient_system_prompts.get(
                    getattr(patient_model, 'name', f'patient_{i}')
                )
            elif isinstance(patient_system_prompts, list) and i < len(patient_system_prompts):
                patient_system_prompt = patient_system_prompts[i]
            
            generation = (context or {}).get("patient_generation", {}).get(i)
            patient = Patient(
                patient_model, system_prompt=patient_system_prompt, enable_cot=enable_patient_cot,
                generation=generation,
                frame_prompts=(context or {}).get("patient_prompt_framing", True) is not False,
                interview_mode=True,
            )
            patients.append(patient)
            
            # Get patient info from context if available, otherwise try to infer from model
            if patient_info_from_context and i < len(patient_info_from_context):
                patient_info.append(patient_info_from_context[i])
            else:
                # Fallback: try to get from model attributes or use defaults
                patient_info.append({
                    "id": i,
                    "provider": getattr(patient_model, 'provider', 'unknown'),
                    "model": getattr(patient_model, 'name', f'patient_{i}'),
                })
        
        # Enable dynamic strategies by default, but allow override from context
        use_dynamic_strategies = context.get("use_dynamic_strategies", True) if context else True
        doctor = Doctor(
            doctor_model, 
            system_prompt=final_doctor_prompt, 
            enable_cot=enable_doctor_cot,
            use_dynamic_strategies=use_dynamic_strategies
        ) if doctor_model else None
        
        conversation_history: List[Dict[str, str]] = []
        
        # Get callback to save turns incrementally if available
        save_turn_callback = context.get("save_conversation_turn_callback") if context else None
        
        # Initialize conversation
        if doctor:
            # Doctor starts the conversation with a question to all patients
            doctor_response = await doctor.conduct_interview("", turn_number=0, context=context)
            doctor_question = _extract_doctor_question(doctor_response.content)
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
            
            # All patients respond to the doctor's question (use extracted question only)
            for i, patient in enumerate(patients):
                patient.conversation_history = _history_for_patient(conversation_history, i)
                patient_response = await patient.respond(
                    "Respond to the doctor's latest question.", context=context,
                )
                reasoning, reasoning_source = reasoning_fields(patient_response)
                
                patient_name = f"Patient {i+1} ({patient_info[i]['model']})"
                turn_data = {
                    "speaker": "patient",
                    **response_turn_fields(patient_response),
                    "patient_id": i,
                    "patient_name": patient_name,
                    "patient_provider": patient_info[i]["provider"],
                    "patient_model": patient_info[i]["model"],
                    "prompt": doctor_question,
                    "response": patient_response.content,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "turn_number": len(conversation_history),
                }
                conversation_history.append(turn_data)
                
                # Save turn immediately if callback available
                if save_turn_callback:
                    await save_turn_callback(turn_data)
                
        
        # Get cancellation check callback if available
        check_cancellation = context.get("check_cancellation") if context else None
        
        # Continue conversation for max_turns
        for turn in range(1, self.max_turns):
            if not doctor:
                break
            
            # Check for cancellation
            if check_cancellation:
                check_cancellation()
            
            # Older rounds are already in the doctor's own history. Add only
            # replies since its latest question so speakers are not replayed.
            last_patient_responses = []
            for turn_item in reversed(conversation_history):
                if turn_item.get("speaker") == "doctor":
                    break
                if turn_item.get("speaker") == "patient" and turn_item.get("patient_id") is not None:
                    last_patient_responses.append(turn_item)
            last_patient_responses.reverse()
            
            # Build summary of patient responses for doctor
            patient_responses_summary = "\n\n".join([
                f"{resp.get('patient_name', 'Patient')}: {resp.get('response', '')}"
                for resp in last_patient_responses
            ])
            
            # Doctor analyzes and asks next question
            doctor_response = await doctor.conduct_interview(
                patient_responses_summary,
                turn_number=turn,
                context=context,
            )
            doctor_question = _extract_doctor_question(doctor_response.content)
            
            # Check for cancellation after doctor response
            if check_cancellation:
                check_cancellation()
            
            reasoning, reasoning_source = reasoning_fields(doctor_response)
            turn_data = {
                "speaker": "doctor",
                **response_turn_fields(doctor_response),
                "prompt": patient_responses_summary,
                "response": doctor_question,
                "reasoning": reasoning,
                "reasoning_source": reasoning_source,
                "turn_number": len(conversation_history),
            }
            conversation_history.append(turn_data)
            
            # Save turn immediately if callback available
            if save_turn_callback:
                await save_turn_callback(turn_data)
            
            # All patients respond to the doctor's new question (use extracted question only)
            for i, patient in enumerate(patients):
                # Check for cancellation
                if check_cancellation:
                    check_cancellation()
                
                patient.conversation_history = _history_for_patient(conversation_history, i)
                patient_response = await patient.respond(
                    "Respond to the doctor's latest question.", context=context,
                )
                reasoning, reasoning_source = reasoning_fields(patient_response)
                
                patient_name = f"Patient {i+1} ({patient_info[i]['model']})"
                turn_data = {
                    "speaker": "patient",
                    **response_turn_fields(patient_response),
                    "patient_id": i,
                    "patient_name": patient_name,
                    "patient_provider": patient_info[i]["provider"],
                    "patient_model": patient_info[i]["model"],
                    "prompt": doctor_question,
                    "response": patient_response.content,
                    "reasoning": reasoning,
                    "reasoning_source": reasoning_source,
                    "turn_number": len(conversation_history),
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
            metadata={
                "conversation_history": conversation_history,
                "patient_info": patient_info,
                **({"doctor_assessment": assessment_record} if assessment_record is not None else {}),
            },
        )
    
    def _summarize_conversation(self, history: List[Dict[str, str]]) -> str:
        """Summarize conversation history."""
        summary = "Group Therapy Session Summary:\n\n"
        for i, turn in enumerate(history, 1):
            speaker = turn.get("speaker", "unknown").title()
            if speaker == "Patient":
                speaker = turn.get("patient_name", "Patient")
            summary += f"Turn {i} ({speaker}): {turn.get('response', '')}\n\n"
        return summary
