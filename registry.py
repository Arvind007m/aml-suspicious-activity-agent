"""
Registry mapping tool names to execution functions and providing validation.
Valid tools:
- eda
- feature_engineering
- anomaly_detection
- risk_classification
- explanation
"""

from typing import Dict, List, Callable, Any

# Valid tool names
VALID_TOOLS = {
    "eda",
    "feature_engineering",
    "anomaly_detection",
    "risk_classification",
    "explanation"
}

# Map of tool names to functions
TOOL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def register_tool(name: str):
    """Decorator to register a tool function."""
    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        if name not in VALID_TOOLS:
            raise ValueError(f"Unknown tool name '{name}'. Must be one of {VALID_TOOLS}")
        TOOL_REGISTRY[name] = func
        return func
    return decorator


def validate_plan(plan: List[str]) -> List[str]:
    """
    Validates tool names in plan.
    Filters out invalid tools and caps iterations at 6.
    """
    validated = [t for t in plan if t in VALID_TOOLS]
    return validated[:6]


def load_all_tools():
    """
    Loads all tool modules so decorators trigger registration.
    """
    import tools.eda
    import tools.feature_engineering
    import tools.anomaly_detection
    import tools.risk_classification
    import tools.explanation


def execute_tool_chain(plan: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes an ordered list of tools sequentially, passing updated context along.
    """
    load_all_tools()
    context["executed_tools"] = []
    
    print("\n" + "="*50)
    print("           EXECUTING AGENT TOOL CHAIN             ")
    print("="*50)
    
    for tool_name in plan:
        if tool_name in TOOL_REGISTRY:
            print(f"  [-> Running Tool]: {tool_name}")
            context = TOOL_REGISTRY[tool_name](context)
            context["executed_tools"].append(tool_name)
        else:
            print(f"  [!] Warning: Tool '{tool_name}' not registered in registry!")
            
    return context
