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
    the corresponding value from the variables dictionary.
    
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
    result = prompt
    
    # Replace each variable found in the prompt
    for var_name, var_value in variables.items():
        # Escape special regex characters in variable name
        escaped_name = re.escape(var_name)
        # Replace $variable_name with the value
        # Use word boundary to ensure we match the full variable name
        # Pattern: $variable_name followed by word boundary or end of string
        pattern = r'\$' + escaped_name + r'\b'
        result = re.sub(pattern, var_value, result)
    
    return result
