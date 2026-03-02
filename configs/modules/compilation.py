from typing import Dict, Any


class CompilationConfig:
    def __init__(self, config_dict: Dict[str, Any]):
        self.config_dict = config_dict
        self.enabled: bool = self._parse_field("enabled", True)
        self.mode: str = self._parse_field("mode", "reduce-overhead")
        self.fullgraph: bool = self._parse_field("fullgraph", True)

    def _parse_field(self, field_name: str, default: Any) -> Any:
        return self.config_dict.get(field_name, default)
