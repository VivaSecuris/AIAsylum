"""Configuration dataclass for comparison runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Any


def _parse_patch_layers(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        return [int(x.strip()) for x in str(value).split(",") if x.strip() != ""]
    except ValueError:
        return None


def _parse_patch_positions(value: Any) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        return [int(x.strip()) for x in str(value).split(",") if x.strip() != ""]
    except ValueError:
        return None


def _parse_patch_heads(value: Any) -> Optional[List[Tuple[int, int]]]:
    if value is None:
        return None
    try:
        parts = [int(x.strip()) for x in str(value).split(",") if x.strip() != ""]
        if len(parts) % 2 == 0:
            return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
    except ValueError:
        pass
    return None


def _parse_patch_neurons(value: Any) -> Optional[List[Tuple[int, int]]]:
    if value is None:
        return None
    try:
        parts = [int(x.strip()) for x in str(value).split(",") if x.strip() != ""]
        if len(parts) % 2 == 0:
            return [(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
    except ValueError:
        pass
    return None


@dataclass
class Config:
    """Configuration for a comparison run."""
    
    # Required (for single mode only prompt_a is used; for comparison both prompt_a and prompt_b are required)
    model: str
    out_dir: Path
    prompt_a: str = ""
    prompt_b: str = ""
    
    # Optional prompts (for library)
    prompt_a_id: Optional[str] = None
    prompt_b_id: Optional[str] = None

    # Single-prompt analysis (analysis_mode == "single")
    prompt: Optional[str] = None
    prompt_id: Optional[str] = None
    
    # Multi-prompt analysis (for one-shot/multi-shot)
    prompts: Optional[List[str]] = None  # For N-prompt analysis
    analysis_mode: str = "comparison"  # "single" (1 prompt), "comparison" (2 prompts), or "progression" (N prompts)
    query_marker: Optional[str] = None  # Marker to separate examples from query (e.g., "<<<QUERY>>>")
    
    # Model configuration
    device: str = "cpu"
    dtype: str = "float32"
    max_len: int = 2048
    seed: int = 0
    
    # Alignment
    align: str = "prefix"
    marker: str = "<<<QUERY>>>"
    window: int = 256
    
    # Visualization
    dim_reduction: str = "pca"  # "pca", "umap", or "tsne"
    pca_layers: str = "auto"
    topk: int = 10
    label_points: int = 20
    verbose: bool = False
    
    # Phase 2: Component localization
    enable_component_analysis: bool = True  # Enable attention/MLP/circuit analysis
    enable_attention_capture: bool = True  # Capture attention weights (requires debug_mode)
    enable_mlp_capture: bool = True  # Capture MLP activations (requires debug_mode)
    enable_attn_output_capture: bool = True  # Capture per-layer attention output (for residual decomposition)
    enable_pre_mlp_capture: bool = False  # Capture pre-MLP intermediate activations (heavier memory)
    enable_qkv_capture: bool = False  # Capture Q,K,V per layer (heavier memory; for OV/QK)

    # PCA parameters (only used when dim_reduction == "pca")
    pca_n_components: int = 3
    pca_random_state: int = 0

    # UMAP parameters (only used when dim_reduction == "umap")
    umap_n_components: int = 3
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    umap_random_state: int = 0

    # t-SNE parameters (only used when dim_reduction == "tsne")
    tsne_n_components: int = 3
    tsne_perplexity: float = 30.0
    tsne_learning_rate: str = "auto"  # Can be "auto" or a float
    tsne_n_iter: int = 1000
    tsne_metric: str = "euclidean"
    tsne_random_state: int = 0
    
    # Activation patching
    enable_patching: bool = False
    patch_layers: Optional[List[int]] = None
    patch_positions: Optional[List[int]] = None  # None = last token only; "all" parsed as range(window_len)
    patch_mode: str = "last_token"  # "last_token" | "multi_position"
    patch_components: Optional[str] = None  # "layer" | "head" | "neuron" for granular patching
    patch_heads: Optional[List[Tuple[int, int]]] = None  # [(layer, head_idx), ...]
    patch_neurons: Optional[List[Tuple[int, int]]] = None  # [(layer, neuron_idx), ...]
    # Causal scrubbing and minimal circuit
    enable_scrub: bool = False
    enable_minimal_circuit: bool = False
    
    # COT analysis
    enable_cot_detection: Optional[bool] = None  # None = auto-detect
    cot_analysis_mode: str = "none"  # none, basic, full
    cot_visualization: bool = True
    
    # Prompt library
    library_path: Optional[Path] = None
    save_prompts: bool = False
    save_prompt_ids: Optional[List[str]] = None
    
    # Debug
    debug_mode: bool = False

    @classmethod
    def from_cli_args(
        cls,
        args: Any,
        resolved_prompt_a: Optional[str] = None,
        resolved_prompts: Optional[List[str]] = None,
    ) -> "Config":
        """Build Config from CLI argparse namespace. Use resolved_prompt_a for single mode, resolved_prompts for progression (after library resolution)."""
        mode = getattr(args, "analysis_mode", "comparison")
        out_dir = getattr(args, "out_dir", Path("out"))
        if not isinstance(out_dir, Path):
            out_dir = Path(out_dir)

        common = dict(
            model=args.model,
            out_dir=out_dir,
            device=getattr(args, "device", "cpu"),
            dtype=getattr(args, "dtype", "float32"),
            max_len=getattr(args, "max_len", 2048),
            seed=getattr(args, "seed", 0),
            dim_reduction=getattr(args, "dim_reduction", "pca"),
            pca_layers=getattr(args, "pca_layers", "auto"),
            topk=getattr(args, "topk", 10),
            enable_component_analysis=True,
            enable_attention_capture=True,
            enable_mlp_capture=True,
            enable_attn_output_capture=not getattr(args, "no_attn_output_capture", False),
            enable_pre_mlp_capture=getattr(args, "enable_pre_mlp_capture", False),
            enable_qkv_capture=getattr(args, "enable_qkv_capture", False),
        )

        if mode == "single":
            prompt = resolved_prompt_a or getattr(args, "prompt_single", None) or getattr(args, "prompt_a", None) or ""
            return cls(
                prompt_a=prompt,
                prompt_b="",
                prompt=prompt,
                prompt_id=getattr(args, "prompt_a_id", None),
                analysis_mode="single",
                window=getattr(args, "window", 256),
                query_marker=getattr(args, "query_marker", None),
                enable_scrub=getattr(args, "enable_scrub", False),
                **common,
            )
        if mode == "progression":
            prompts = resolved_prompts or getattr(args, "prompts", None) or []
            return cls(
                prompt_a="",
                prompt_b="",
                analysis_mode="progression",
                prompts=prompts,
                query_marker=getattr(args, "query_marker", None),
                align=getattr(args, "align", "prefix"),
                marker=getattr(args, "marker", "<<<QUERY>>>"),
                window=getattr(args, "window", 256),
                **common,
            )
        # comparison
        patch_layers = _parse_patch_layers(getattr(args, "patch_layers", None))
        patch_positions = _parse_patch_positions(getattr(args, "patch_positions", None))
        patch_heads = _parse_patch_heads(getattr(args, "patch_heads", None))
        patch_neurons = _parse_patch_neurons(getattr(args, "patch_neurons", None))
        return cls(
            prompt_a=getattr(args, "prompt_a", ""),
            prompt_b=getattr(args, "prompt_b", ""),
            analysis_mode="comparison",
            align=getattr(args, "align", "prefix"),
            marker=getattr(args, "marker", "<<<QUERY>>>"),
            window=getattr(args, "window", 256),
            label_points=getattr(args, "label_points", 20),
            verbose=getattr(args, "verbose", False),
            enable_patching=getattr(args, "enable_patching", False),
            patch_layers=patch_layers,
            patch_positions=patch_positions,
            patch_mode=getattr(args, "patch_mode", "last_token"),
            patch_components=getattr(args, "patch_components", None),
            patch_heads=patch_heads,
            patch_neurons=patch_neurons,
            enable_scrub=getattr(args, "enable_scrub", False),
            enable_minimal_circuit=getattr(args, "enable_minimal_circuit", False),
            **common,
        )

    @classmethod
    def from_api_request(cls, request: Any, out_dir: Path) -> "Config":
        """Build Config from server API request (e.g. ComparisonRequest). out_dir is typically Path(f'/tmp/vivasecuris.aiasylum.interp/{job_id}')."""
        analysis_mode = getattr(request, "analysis_mode", None) or "single"
        if getattr(request, "prompts", None) and len(request.prompts) >= 2:
            analysis_mode = "progression"
        elif getattr(request, "analysis_mode", None) == "single":
            analysis_mode = "single"
        elif analysis_mode != "progression":
            analysis_mode = "comparison"

        common = dict(
            model=request.model,
            out_dir=out_dir,
            device=getattr(request, "device", "cpu"),
            dtype=getattr(request, "dtype", "float32"),
            max_len=2048,
            seed=0,
            dim_reduction=getattr(request, "dim_reduction", "pca"),
            pca_layers=getattr(request, "pca_layers", "auto"),
            topk=getattr(request, "topk", 10),
            enable_component_analysis=getattr(request, "enable_component_analysis", True),
            enable_attention_capture=True,
            enable_mlp_capture=True,
            enable_attn_output_capture=True,
            cot_analysis_mode=getattr(request, "cot_analysis_mode", "none"),
        )

        if analysis_mode == "single":
            single_prompt = getattr(request, "prompt", None) or getattr(request, "prompt_a", None) or ""
            return cls(
                prompt_a=single_prompt,
                prompt_b="",
                prompt=single_prompt,
                prompt_id=getattr(request, "prompt_id", None),
                analysis_mode="single",
                window=getattr(request, "window", 256),
                query_marker=getattr(request, "query_marker", None),
                **common,
            )
        if analysis_mode == "progression":
            return cls(
                prompt_a="",
                prompt_b="",
                analysis_mode="progression",
                prompts=getattr(request, "prompts", None) or [],
                query_marker=getattr(request, "query_marker", None),
                align=getattr(request, "align", "prefix"),
                marker=getattr(request, "marker", None) or "<<<QUERY>>>",
                window=getattr(request, "window", 256),
                **common,
            )
        return cls(
            prompt_a=getattr(request, "prompt_a", ""),
            prompt_b=getattr(request, "prompt_b", ""),
            analysis_mode="comparison",
            align=getattr(request, "align", "prefix"),
            marker=getattr(request, "marker", None) or "<<<QUERY>>>",
            window=getattr(request, "window", 256),
            enable_patching=getattr(request, "enable_patching", False),
            patch_layers=getattr(request, "patch_layers", None),
            patch_positions=getattr(request, "patch_positions", None),
            patch_mode=getattr(request, "patch_mode", "last_token"),
            **common,
        )