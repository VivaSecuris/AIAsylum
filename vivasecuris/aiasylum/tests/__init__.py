"""Test framework and test cases."""

from vivasecuris.aiasylum.tests.base import TestCase, TestResult as TestResultType
from vivasecuris.aiasylum.tests.one_shot import OneShotTest
from vivasecuris.aiasylum.tests.multi_shot import MultiShotTest
# Legacy tests (kept for backward compatibility)
from vivasecuris.aiasylum.tests.conversation import ConversationTest
from vivasecuris.aiasylum.tests.scenario import ScenarioTest
from vivasecuris.aiasylum.tests.adversarial import AdversarialTest

__all__ = [
    "TestCase",
    "TestResultType",
    "OneShotTest",
    "MultiShotTest",
    "ConversationTest",  # Legacy
    "ScenarioTest",  # Legacy
    "AdversarialTest",  # Legacy
]
