"""Utility functions for handling variables in prompts."""

import re
from typing import Dict, List


def extract_variables(prompt: str) -> List[str]:
    """
    Extract variable names from a prompt.
    
    Variables are identified by the pattern $variable_name where variable_name
    can contain letters, numbers, and underscores.
    
    Args:
        prompt: The prompt text to extract variables from
        
    Returns:
        List of unique variable names (without the $ prefix), sorted alphabetically
        
    Examples:
        >>> extract_variables("Please tell me the capital of $country")
        ['country']
        >>> extract_variables("Hello $name, you are $age years old")
        ['age', 'name']
        >>> extract_variables("No variables here")
        []
    """
    # Pattern matches $ followed by one or more word characters (letters, digits, underscore)
    # The pattern uses word boundaries to avoid matching $ in the middle of words
    pattern = r'\$(\w+)'
    matches = re.findall(pattern, prompt)
    
    # Return unique variables, sorted
    return sorted(list(set(matches)))


def substitute_variables(prompt: str, variables: Dict[str, str]) -> str:
    """
    Substitute variables in a prompt with their values.
    
    Variables are identified by the pattern $variable_name and replaced with
    the corresponding literal value from the variables dictionary. Inserted
    values are not interpreted as regex replacements or expanded a second time.
    
    Args:
        prompt: The prompt text containing variables
        variables: Dictionary mapping variable names (without $) to their values
        
    Returns:
        The prompt with variables substituted
        
    Examples:
        >>> substitute_variables("Capital of $country", {"country": "France"})
        'Capital of France'
        >>> substitute_variables("Hello $name", {"name": "Alice"})
        'Hello Alice'
        >>> substitute_variables("No variables", {})
        'No variables'
    """
    if not variables:
        return prompt

    # Keep the existing literal-name and trailing word-boundary matching rules.
    # One pass prevents an inserted $name from becoming another substitution;
    # a callback prevents paths/backreferences from being regex instructions.
    names = "|".join(re.escape(name) for name in variables)
    return re.sub(r"\$(" + names + r")\b", lambda match: variables[match.group(1)], prompt)
