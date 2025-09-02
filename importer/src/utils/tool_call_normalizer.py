from typing import Any, Dict, List, Optional


class ToolCallNormalizer:
    """Normalizes tool call parameters to ensure consistency."""

    @staticmethod
    def extract_name(value: Any) -> Any:
        """Extract name from dict or return value as-is."""
        return value.get("name") if isinstance(value, dict) and "name" in value else value

    @classmethod
    def extract_relation_entity_names(cls, relation: Dict[str, Any]) -> List[str]:
        """Extract entity names from a relation dict (source and target)."""
        names = []
        if isinstance(relation, dict):
            source = relation.get("source")
            target = relation.get("target")

            if isinstance(source, dict):
                source = source.get("name")
            if isinstance(target, dict):
                target = target.get("name")

            if isinstance(source, str):
                names.append(source)
            if isinstance(target, str):
                names.append(target)
        return names

    @classmethod
    def normalize_single_relation(cls, relation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single relation dictionary."""
        normalized = dict(relation)

        # Handle field name variations
        if "subject" in normalized and "source" not in normalized:
            normalized["source"] = cls.extract_name(normalized.pop("subject"))
        if "object" in normalized and "target" not in normalized:
            normalized["target"] = cls.extract_name(normalized.pop("object"))
        if "predicate" in normalized and "relationType" not in normalized:
            normalized["relationType"] = normalized.pop("predicate")

        normalized["source"] = cls.extract_name(normalized.get("source"))
        normalized["target"] = cls.extract_name(normalized.get("target"))
        return normalized

    @classmethod
    def normalize_relations_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for create_relations tool."""
        relations = params.get("relations")

        if not relations:
            # Try to build from individual fields
            maybe = {k: params.get(k) for k in ("source", "predicate", "relationType", "target", "when", "evidence", "confidence", "sourceId", "chunkId", "sourceUrl") if k in params}
            if any(maybe.values()):
                relations = [maybe]

        normalized_relations = [
            cls.normalize_single_relation(r)
            for r in relations or []
            if isinstance(r, dict)
        ]

        result = {k: v for k, v in params.items() if k != "relations"}
        result["relations"] = normalized_relations
        return result

    @classmethod
    def normalize_observation(cls, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single observation dictionary."""
        normalized = dict(observation)

        # Normalize entity name field
        entity_name_mappings = [
            ("entity", "entityName"),
            ("entity_name", "entityName"),
            ("name", "entityName")
        ]

        for old_key, new_key in entity_name_mappings:
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = cls.extract_name(normalized.pop(old_key))

        # Normalize observations field
        if "observations" not in normalized or not isinstance(normalized.get("observations"), list):
            if "text" in normalized and normalized["text"]:
                normalized["observations"] = [str(normalized.pop("text"))]
            else:
                normalized["observations"] = []

        return normalized

    @classmethod
    def extract_observations_list(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the observation list from parameters."""
        if observations := params.get("observations"):
            return observations

        if "observation" in params and isinstance(params["observation"], dict):
            return [params["observation"]]

        if any(k in params for k in ("entityName", "text", "observations")):
            observations_list = cls.build_observations_list(params)
            return [{"entityName": params.get("entityName"), "observations": observations_list}]

        return []

    @classmethod
    def build_observations_list(cls, params: Dict[str, Any]) -> List[str]:
        """Build the observation list from parameters."""
        if isinstance(params.get("observations"), list):
            return [str(x) for x in params["observations"]]
        elif "text" in params and params["text"]:
            return [str(params["text"])]
        return []

    @classmethod
    def normalize_observations_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for the add_observations tool."""
        observations = cls.extract_observations_list(params)
        normalized_observations = [cls.normalize_observation(o) for o in observations or [] if isinstance(o, dict)]

        result = {k: v for k, v in params.items() if k not in {"observations", "observation", "text"}}
        result["observations"] = normalized_observations
        return result

    @classmethod
    def normalize_single_entity(cls, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a single entity dictionary."""
        normalized = {k: v for k, v in entity.items() if k in {"name", "type", "observations"}}

        if "name" not in normalized:
            return None

        if "observations" not in normalized or not isinstance(normalized.get("observations"), list):
            normalized["observations"] = []

        return normalized

    @classmethod
    def extract_entities_from_params(cls, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities list from various parameter formats."""
        if entities := params.get("entities"):
            return entities

        if "entity" in params and isinstance(params["entity"], dict):
            return [params["entity"]]

        if any(k in params for k in ("name", "type")):
            entity = {k: params.get(k) for k in ("name", "type") if k in params}
            return [entity]

        if cls.has_valid_observations(params):
            first_observation = params["observations"][0]
            entity = {
                "name": first_observation["entityName"], 
                "type": params.get("type", "Thing")
            }
            return [entity]

        return []

    @classmethod
    def has_valid_observations(cls, params: Dict[str, Any]) -> bool:
        """Check if parameters contain valid observations with an entity name."""
        observations = params.get("observations")
        if not isinstance(observations, list) or not observations:
            return False

        first_observation = observations[0]
        return isinstance(first_observation, dict) and "entityName" in first_observation

    @classmethod
    def normalize_entities_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for the create_entities tool."""
        entities = cls.extract_entities_from_params(params)
        normalized_entities = [
            normalized_entity 
            for entity in entities or []
            if isinstance(entity, dict) and (normalized_entity := cls.normalize_single_entity(entity))
        ]

        result = {k: v for k, v in params.items() if k != "entities"}
        result["entities"] = normalized_entities
        return result

    @classmethod
    def normalize_params(cls, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize parameters for different tools."""
        normalizers = {
            "create_relations": cls.normalize_relations_params,
            "add_observations": cls.normalize_observations_params,
            "create_entities": cls.normalize_entities_params,
        }

        normalizer = normalizers.get(tool_name)
        return normalizer(params) if normalizer else params
