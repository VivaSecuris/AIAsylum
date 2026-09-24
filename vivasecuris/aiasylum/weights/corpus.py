"""Harmful / harmless prompt pairs for direction derivation.

Harmful prompts come from aiasylum's own ``PromptLibrary`` table, category
``forbidden_question`` -- direct harmful requests. The ``adversarial`` category
is deliberately *not* used here: those are jailbreak wrapper templates (DAN
personas and the like), which belong in evaluation rather than in deriving what
refusal looks like in the residual stream.

That table holds the 13-scenario forbidden-question set, and five of those
scenarios are *policy-restricted advice* rather than harm -- legal opinions,
financial advice, health consultations, government decisions, political
lobbying. Models answer those freely, usually with a disclaimer. Including them
puts benign prompts in the "harmful" class and blunts the difference-in-means
badly, so only the eight refusal-triggering scenarios are used by default.

Everything is split train/test with a fixed seed and the split is hashed. The
direction is fit on train only and every number reported comes from test;
without that, a strong result may only mean the direction memorized its own
fitting set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Scenarios that actually elicit refusal from a safety-tuned model.
HARMFUL_SCENARIOS = (
    "Illegal Activity",
    "Hate Speech",
    "Malware",
    "Physical Harm",
    "Fraud",
    "Pornography",
    "Privacy Violence",
    "Economic Harm",
)

# Present in the same table but answered rather than refused; excluded by
# default so they do not dilute the harmful class.
ADVICE_SCENARIOS = (
    "Legal Opinion",
    "Financial Advice",
    "Health Consultation",
    "Gov Decision",
    "Political Lobbying",
)

# Neutral requests matched to the harmful set in form -- imperative or
# interrogative task requests -- so the difference-in-means isolates harmfulness
# rather than sentence mood. Used when no benchmark dataset is available.
FALLBACK_HARMLESS = [
    "How can I improve my resume for a software engineering role?",
    "What is the recipe for a classic French omelette?",
    "Can you help me set up a home network with a guest VLAN?",
    "How can I grow tomatoes in a small apartment balcony?",
    "What is the best way to learn conversational Spanish?",
    "How do I change a flat tire on a bicycle?",
    "Can you explain how a refrigerator keeps food cold?",
    "What should I look for when adopting a rescue dog?",
    "How can I reduce my monthly electricity bill?",
    "What is the difference between a virus and a bacterium?",
    "How do I back up my photos to an external drive?",
    "Can you suggest a beginner strength training routine?",
    "What causes the northern lights?",
    "How can I make my apartment more energy efficient?",
    "What is the proper way to season a cast iron skillet?",
    "How do I read a nutrition label?",
    "Can you explain how compound interest works?",
    "What are good practices for writing unit tests?",
    "How do I prune an apple tree?",
    "What is the history of the printing press?",
    "How can I improve my sleep quality?",
    "What is the best way to remove a coffee stain from a shirt?",
    "Can you explain how vaccines train the immune system?",
    "How do I plan a two week trip to Japan?",
    "What is the difference between weather and climate?",
    "How can I start composting at home?",
    "What should I consider when buying a used car?",
    "How do I tune a guitar by ear?",
    "Can you explain how GPS determines location?",
    "What are effective techniques for public speaking?",
    "How do I make sourdough starter from scratch?",
    "What is the function of the mitochondria in a cell?",
    "How do I set up a recurring backup on a Synology NAS?",
    "What is the best way to store fresh basil so it lasts?",
    "Can you explain the difference between TCP and UDP?",
    "How do I replace the brake pads on a mountain bike?",
    "What should I pack for a week of winter camping?",
    "How can I teach my kid to ride a bicycle?",
    "What is the difference between baking soda and baking powder?",
    "Can you explain how noise cancelling headphones work?",
    "How do I get rid of fruit flies in my kitchen?",
    "What is a good beginner telescope for viewing planets?",
    "How do I write a cover letter for a career change?",
    "What causes bread dough to fail to rise?",
    "Can you explain how a heat pump differs from a furnace?",
    "How do I clean a laptop keyboard safely?",
    "What are the rules of cricket in simple terms?",
    "How can I improve the drainage in a clay-heavy garden bed?",
    "What is the difference between espresso and drip coffee?",
    "Can you explain what a blockchain actually stores?",
    "How do I train a puppy to stop jumping on guests?",
    "What is the safest way to jump start a car battery?",
    "How do I choose between renting and buying a home?",
    "What is the difference between RAM and storage?",
    "Can you explain how tides are caused by the moon?",
    "How do I sharpen a kitchen knife with a whetstone?",
    "What are good stretches for lower back stiffness?",
    "How do I set up a drip irrigation system for raised beds?",
    "What is the history of the Silk Road trade routes?",
    "Can you explain how an electric guitar pickup works?",
    "How do I remove wallpaper without damaging the drywall?",
    "What is the difference between a symphony and a concerto?",
    "How can I make my resume pass automated screening?",
    "What is the proper technique for a deadlift?",
    "Can you explain how DNS resolves a domain name?",
    "How do I keep squirrels out of a bird feeder?",
    "What is the difference between a mortgage rate and APR?",
    "How do I start learning to play chess seriously?",
    "What causes the different colors in autumn leaves?",
    "Can you explain how a car transmission works?",
    "How do I fix a running toilet?",
    "What is the best way to learn touch typing?",
    "How do I build a simple bookshelf from plywood?",
    "What is the difference between weather radar and satellite?",
    "Can you explain how solar panels convert light to electricity?",
    "How do I plan meals for a week on a tight budget?",
    "What are the basic rules of Go?",
    "How do I photograph the night sky with a DSLR?",
    "What is the difference between a virus and a worm in software?",
    "Can you explain how a sourdough culture stays alive?",
    "How do I winterize an outdoor faucet?",
    "What is the best way to memorize vocabulary in a new language?",
    "How do I choose running shoes for flat feet?",
    "What causes an engine to overheat?",
    "Can you explain how libraries decide which books to stock?",
    "How do I set up parental controls on a home router?",
    "What is the difference between cement and concrete?",
    "How do I make cold brew coffee at home?",
    "What are effective ways to prepare for a job interview?",
    "Can you explain how airplanes generate lift?",
    "How do I repair a small tear in a down jacket?",
    "What is the difference between a debit and a credit card?",
    "How do I grow herbs indoors during winter?",
    "What is the best way to organize a garage?",
    "Can you explain how a microwave heats food?",
    "How do I teach myself basic music theory?",
    "What is the difference between hardwood and laminate flooring?",
    "How do I plan a budget for a kitchen renovation?",
    "What causes static electricity in dry weather?",
    "Can you explain how a bicycle derailleur shifts gears?",
    "How do I get better at reading financial statements?",
    "What is the proper way to store wine long term?",
    "How do I identify edible mushrooms safely with a guide?",
    "What is the difference between an alloy and a compound?",
    "Can you explain how rechargeable batteries degrade?",
    "How do I set up a compost bin in a small yard?",
    "What are good techniques for speed reading?",
    "How do I choose a mattress for side sleeping?",
    "What is the difference between a hurricane and a typhoon?",
    "Can you explain how a thermostat controls temperature?",
    "How do I plan a vegetable garden crop rotation?",
    "What is the best way to learn to swim as an adult?",
    "How do I remove hard water stains from a shower door?",
    "What is the difference between a telescope and binoculars?",
    "Can you explain how sound travels through different materials?",
    "How do I keep houseplants alive while traveling?",
    "What are the basics of good photographic composition?",
    "How do I replace a light switch safely?",
    "What is the difference between whole and skim milk nutritionally?",
    "Can you explain how a dishwasher cleans dishes?",
    "How do I prepare a lawn for overseeding?",
    "What is the best way to track personal expenses?",
    "How do I learn to read sheet music?",
    "What causes a computer to slow down over time?",
    "Can you explain how bridges handle expansion in heat?",
    "How do I choose a bike for commuting in a city?",
    "What is the difference between a lease and a rental agreement?",
    "How do I make homemade pasta without a machine?",
]


@dataclass
class PromptSplit:
    """A seeded, hashed train/test split over both classes."""

    harmful_train: List[str]
    harmful_test: List[str]
    harmless_train: List[str]
    harmless_test: List[str]
    seed: int
    source: str

    @property
    def hash(self) -> str:
        """Stable digest of the exact prompts used, for the surgery manifest."""
        payload = json.dumps(
            {
                "harmful_train": self.harmful_train,
                "harmful_test": self.harmful_test,
                "harmless_train": self.harmless_train,
                "harmless_test": self.harmless_test,
                "seed": self.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def summary(self) -> dict:
        return {
            "source": self.source,
            "seed": self.seed,
            "split_hash": self.hash,
            "n_harmful_train": len(self.harmful_train),
            "n_harmful_test": len(self.harmful_test),
            "n_harmless_train": len(self.harmless_train),
            "n_harmless_test": len(self.harmless_test),
        }


def load_harmful_prompts(
    limit: Optional[int] = None,
    category: str = "forbidden_question",
    scenarios: Optional[Sequence[str]] = HARMFUL_SCENARIOS,
) -> List[str]:
    """Load direct harmful requests from aiasylum's PromptLibrary.

    ``scenarios`` filters on the imported ``content_policy_name``. Pass ``None``
    to take every scenario, including the restricted-advice ones -- see this
    module's docstring for why that is not the default.
    """
    from vivasecuris.aiasylum.database.models import PromptLibrary
    from vivasecuris.aiasylum.database.session import get_session

    session = get_session()
    try:
        rows = (
            session.query(PromptLibrary.prompt_text, PromptLibrary.meta_data)
            .filter(PromptLibrary.category == category)
            .order_by(PromptLibrary.id)
            .all()
        )
    finally:
        session.close()

    allowed = set(scenarios) if scenarios else None
    prompts, skipped = [], 0
    for text, meta in rows:
        if not text or not text.strip():
            continue
        if allowed is not None:
            scenario = (meta or {}).get("content_policy_name")
            if scenario is not None and scenario not in allowed:
                skipped += 1
                continue
        prompts.append(text.strip())

    if not prompts:
        raise ValueError(
            f"No prompts found in category '{category}'"
            + (f" for scenarios {sorted(allowed)}" if allowed else "")
            + ". Populate the library first: python scripts/import_jailbreaks.py"
        )

    logger.info(
        "Loaded %d harmful prompts (category=%s, %d excluded as restricted-advice)",
        len(prompts), category, skipped,
    )
    return prompts[:limit] if limit else prompts


def load_harmless_prompts(limit: Optional[int] = None, benchmark: Optional[str] = None) -> List[str]:
    """Load neutral contrast prompts, preferring a cached benchmark.

    Defaults to the built-in set. That set is deliberately form-matched to the
    harmful prompts -- conversational requests to an assistant -- so the
    difference-in-means isolates harmfulness. Academic benchmark questions
    (MMLU and friends) differ from the harmful set in register as well as in
    content, so a direction fit against them partly encodes "academic vs
    conversational" instead. Pass ``benchmark`` explicitly to use one anyway.

    Hub lookups are forced offline: the datasets are cached, and an online
    resolve can block indefinitely with no timeout.
    """
    import asyncio
    import os

    if benchmark is None:
        prompts = list(FALLBACK_HARMLESS)
        logger.info("Using %d built-in form-matched harmless prompts", len(prompts))
        return prompts[:limit] if limit else prompts

    prior = {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from vivasecuris.aiasylum.benchmarks.datasets import load_benchmark_dataset

        # load_benchmark_dataset is a coroutine; this module is sync by design.
        rows = asyncio.run(load_benchmark_dataset(benchmark, num_samples=(limit or 256) * 2))
        prompts = [
            (r.get("question") or "").strip()
            for r in rows
            if isinstance(r, dict) and (r.get("question") or "").strip()
        ]
        # MMLU questions can be long passages; keep the short, request-shaped ones.
        prompts = [p for p in prompts if 20 <= len(p) <= 300]
        if len(prompts) >= 32:
            logger.info("Loaded %d harmless prompts from the %s benchmark", len(prompts), benchmark)
            return prompts[:limit] if limit else prompts
        logger.warning("%s returned only %d usable rows; using fallback set", benchmark, len(prompts))
    except Exception as exc:
        logger.warning("Could not load %s (%s: %s); using fallback set", benchmark, type(exc).__name__, exc)
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    prompts = list(FALLBACK_HARMLESS)
    return prompts[:limit] if limit else prompts


def build_split(
    n_per_class: int = 128,
    test_fraction: float = 0.25,
    seed: int = 0,
    harmful: Optional[List[str]] = None,
    harmless: Optional[List[str]] = None,
) -> PromptSplit:
    """Build a balanced, seeded train/test split over both classes."""
    harmful = harmful if harmful is not None else load_harmful_prompts()
    harmless = harmless if harmless is not None else load_harmless_prompts()

    rng = random.Random(seed)
    harmful = sorted(set(harmful))
    harmless = sorted(set(harmless))
    rng.shuffle(harmful)
    rng.shuffle(harmless)

    # Balanced classes, bounded by whichever pool is smaller.
    n = min(n_per_class, len(harmful), len(harmless))
    if n < 8:
        raise ValueError(
            f"Need at least 8 prompts per class, got {n} "
            f"(harmful={len(harmful)}, harmless={len(harmless)})"
        )
    if n < n_per_class:
        logger.warning("Requested %d per class but only %d available", n_per_class, n)

    harmful, harmless = harmful[:n], harmless[:n]
    n_test = max(2, int(round(n * test_fraction)))

    split = PromptSplit(
        harmful_train=harmful[n_test:],
        harmful_test=harmful[:n_test],
        harmless_train=harmless[n_test:],
        harmless_test=harmless[:n_test],
        seed=seed,
        source=f"prompt_library:forbidden_question + harmless(n={len(harmless)})",
    )
    logger.info("Built split %s: %s", split.hash, split.summary())
    return split
