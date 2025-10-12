"""Prompt templates for the agent orchestrator.

Separated from the main orchestrator to reduce file size and keep prompts
versionable / swappable independently.
"""

from __future__ import annotations

PROMPT_TMPL = """You are a professional linguist and a professional journalist in one person. Below the >>TEXTSTART a text will be embedded and I need to extract key information to build a knowledge base based on widely spreaded information. 

You should understand the text, extract entities like:
EXAMPLE ENTITY TYPES TO LOOK FOR (there could be more...):
- Person: individuals, names, roles, titles
- Organization: companies, institutions, agencies, groups
- Location: cities, countries, addresses, places
- Event: meetings, conferences, incidents, announcements
- Concept: technologies, methodologies, ideas, topics
- Method: algorithms, techniques, processes with or without steps
- Tool: tools, tools with features, products, products with features
- Product: software, services, tools, offerings
- Date: specific dates, time periods, deadlines
- Amount: money, quantities, percentages, metrics

Into each entity some key observations  (like use cases, facts, numbers, prices, owners, web pages, email addresses, addresses, etc.) and the entity type and category needs to be also extracted.

You also need to determine the relations between the existing or newly extracted entities.
MORE RELATIONSHIP TYPES (examples):
- works_for, employed_by, ceo_of, founded_by, owns
- located_in, headquartered_in, based_in, operates_in
- announced, acquired, merged_with, partnered_with
- developed, created, released, launched, offers
- attended, spoke_at, organized, participated_in
- related_to, part_of, depends_on, uses, implements
- provides, needs, costs, contains, requires
Good relation example: entity1 -> efficiently handles -> entity2
Bad relation example: entity1 -> handles -> entity2 efficiently; do not mix the entities with the relation types.
Relation type should be lowercase in the final output.

EXTRACTION STRATEGY:
- Start by searching for key entities, then extract systematically
- Look for proper nouns (capitalized words/phrases)
- Identify relationships indicated by verbs and prepositions
- Extract specific dates, amounts, and measurements
- Note roles, titles, and positions mentioned
- Capture business relationships and hierarchies
- Record events, announcements, technologies, products and activities
- Deletion of existing entity is not allowed
- Since you are a knowledge builder, try to connect the new information to the existing
- START ANSWER: Write "JSONSTART" before each JSON object (or object list) output

The required output: 

  You should follow these rules:
  - You just have the following (and ONLY THE FOLLOWING) external tools with the required parameters to manipulate a knowledge graph database:
    - read_graph: query: str ; for searching memories based on a query containing search terms
    - find_memories_by_name: names[string]
    - create_entities: entities[{ name, type, observations }] where observations is a list of strings (can be empty)
    - create_relations: relations[{ source, relationType, target }]
    - add_observations: observations[{ entityName, observations:str[] }] where observations is a list of strings
  - Evidence entity is a normal entity with the following properties and observations: name="Evidence {SOURCE_ID}-{CHUNK_ID}", type="Evidence", observations=["srcId={SOURCE_ID}", "chunk={CHUNK_ID}", "url={SOURCE_URL}"]
  - Create Evidence entity: "Evidence {SOURCE_ID}-{CHUNK_ID}" based on the provided document-id and chunk-id on the first lines of the text
  - Link all newly observed entities to Evidence with "evidence" relations (use create_relations tool)
  - Search for existing entities using the relevant tools (find_memories_by_name, read_graph, search_memories) before creating new ones and update the observations only when found.
  - Create meaningful relationships between entities
  - Add rich observations with context and details
  - Use canonical names (e.g., "John Smith" not "Mr. Smith")
  - Be thorough - extract as much structured knowledge as possible
  - If the new relations contains entities as targets which are not created by you, you must create them with a proper Category and with Type=Placeholder.
  - The required output should be ONLY a json formatted, machine-readable JSON object with an array (without any unformatted or irrelevant repetition in other format) of tool calls in a proper order (I mean: create the evidence entity, add_observations first, then create_entites including the observations, then create_relations) using this schema below (method is always "tools/call", you should fill the THE_TOOL_NAME and THE_REQUIRED_TOOL_PARAMETERS_AS_DEFINED_ABOVE):
    {[{"method": "tools/call", "params": {"tool_name": "<THE_TOOL_NAME>", "arguments": {REQUIRED_TOOL_PARAMETERS_AS_DEFINED_ABOVE}}},...]}

VERY IMPORTANT WORKFLOW RULES, MUST BE ALWAYS FOLLOWED:
  - CREATE THE EVIDENCE ENTITY FIRST
  - CALL add_observations before any create_entities to ensure the UPSERT workflow
  - CALL create_entities FOR THE EXTRACTED ENTITIES ONLY AFTER THE add_observations calls
  - create the relations between the new entitles if any, every new entity MUST have at least 2 relations including the Evidence.
  - when creating a relation ensure both end is created by you or was existing before. You might repeat creation of an entity if needed, this could be handled by the graph database engine but when you define a relation without existing both of source and target entities is not allowed.
  - Duplicated entity is not a problem but missing relation is a serious problem.
  - link new entities to the created evidence entity using relation type 'evidence' and tool: create_relations
  - DO NOT OUTPUT ANYTHING ELSE EXCLUDING THE JSON OUTPUT AND USE ONLY THE PROVIDED TOOLS, THERE IS NO 'link_to_evidence' or 'create_link_to_evidence' or 'link_entity_to_evidence' tools, only 'create_relations' as I mentioned above.
  - ENSURE EVERY NEW ENTITY HAVE AT LEAST TWO RELATIONS INCLUDING THE EVIDENCE
  - ALL NEW ENTITY MUST BE LINKED TO ITS EVIDENCE

>>TEXTSTART
{TEXT}
"""

__all__ = ["PROMPT_TMPL"]
