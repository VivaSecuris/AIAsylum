"""Single entry point for running analysis by mode (single, comparison, progression)."""

import logging
from pathlib import Path
from typing import Optional, Union, Any, Tuple

from vivasecuris.aiasylum.interp.core.config import Config
from vivasecuris.aiasylum.interp.core.loader import ModelLoader
from vivasecuris.aiasylum.interp.core.services.comparison_service import ComparisonService
from vivasecuris.aiasylum.interp.core.services.single_prompt_service import SinglePromptService
from vivasecuris.aiasylum.interp.core.services.multi_prompt_service import MultiPromptService
from vivasecuris.aiasylum.interp.visualization.dashboard_builder import DashboardBuilder
from vivasecuris.aiasylum.interp.visualization.single_prompt_dashboard_builder import SinglePromptDashboardBuilder
from vivasecuris.aiasylum.interp.visualization.progression_dashboard_builder import ProgressionDashboardBuilder
from vivasecuris.aiasylum.interp.data.models import ComparisonResult, SinglePromptResult
from vivasecuris.aiasylum.interp.data.multi_prompt_models import ProgressionResult

logger = logging.getLogger(__name__)

ResultT = Union[SinglePromptResult, ComparisonResult, ProgressionResult]


class AnalysisOrchestrator:
    """Runs analysis by config.analysis_mode and optionally saves results and builds dashboard."""

    @staticmethod
    def run(
        config: Config,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> ResultT:
        """
        Run analysis for the configured mode. Loads model if not provided.

        Returns:
            SinglePromptResult, ComparisonResult, or ProgressionResult depending on config.analysis_mode.
        """
        if model is None or tokenizer is None:
            logger.info("Loading model: %s", config.model)
            model, tokenizer = ModelLoader.load_model(
                config.model,
                device=config.device,
                dtype=config.dtype,
                max_length=config.max_len,
            )
        else:
            ModelLoader.set_deterministic(config.seed)

        mode = config.analysis_mode
        if mode == "single":
            return SinglePromptService.run_single_analysis(model, tokenizer, config)
        if mode == "progression":
            return MultiPromptService.run_progression_analysis(model, tokenizer, config)
        if mode == "comparison":
            return ComparisonService.run_comparison(model, tokenizer, config)
        raise ValueError(f"Unknown analysis_mode: {mode}")

    @staticmethod
    def save_results(result: ResultT, out_dir: Path) -> None:
        """Save result JSON and artifacts to out_dir."""
        if isinstance(result, SinglePromptResult):
            SinglePromptService.save_results(result, out_dir)
        elif isinstance(result, ComparisonResult):
            ComparisonService.save_results(result, out_dir)
        elif isinstance(result, ProgressionResult):
            MultiPromptService.save_results(result, out_dir)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")

    @staticmethod
    def build_dashboard(result: ResultT) -> str:
        """Return HTML dashboard for the given result."""
        if isinstance(result, SinglePromptResult):
            return SinglePromptDashboardBuilder().build_dashboard(result)
        if isinstance(result, ComparisonResult):
            return DashboardBuilder().build_dashboard(result)
        if isinstance(result, ProgressionResult):
            return ProgressionDashboardBuilder().build_dashboard(result)
        raise TypeError(f"Unexpected result type: {type(result)}")

    @classmethod
    def run_and_save(
        cls,
        config: Config,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Tuple[ResultT, str]:
        """
        Run analysis, save results to config.out_dir, and write dashboard.html.
        Returns (result, dashboard_html).
        """
        result = cls.run(config, model=model, tokenizer=tokenizer)
        cls.save_results(result, config.out_dir)
        logger.info("Generating dashboard")
        dashboard_html = cls.build_dashboard(result)
        (config.out_dir / "dashboard.html").write_text(dashboard_html)
        return result, dashboard_html
