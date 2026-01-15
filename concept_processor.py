"""
SAAAM Language - Concept Processor
THE BREAKTHROUGH: No tokenization. Process CONCEPTS directly like SAM.

This is where SAAAM becomes revolutionary - we don't tokenize syntax,
we process MEANING and INTENT directly from source code.
"""

import re
import math
import random
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

class ConceptType(Enum):
    """Types of concepts we can extract from source."""
    # Core concepts
    IDENTITY = auto()        # Variables, functions, types
    TRANSFORM = auto()       # Operations, morphing
    FLOW = auto()           # Control flow, data flow  
    STRUCTURE = auto()      # Data structures, nesting
    NEURAL = auto()         # Neuroplastic behavior
    REACTIVE = auto()       # Component reactivity
    
    # Semantic concepts
    INTENT = auto()         # What the programmer intends
    RELATIONSHIP = auto()   # How concepts relate
    PATTERN = auto()        # Recurring structures
    ABSTRACTION = auto()    # Higher-level concepts

@dataclass
class Concept:
    """A concept extracted from source code."""
    concept_type: ConceptType
    content: str
    semantic_vector: List[float]  # Deterministic embedding (stdlib-only)
    location: Tuple[int, int]    # (line, col)
    confidence: float = 1.0
    relationships: List['ConceptRelation'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ConceptRelation:
    """Relationship between concepts."""
    target: 'Concept'
    relation_type: str  # 'contains', 'transforms_to', 'flows_to', 'depends_on'
    strength: float = 1.0

class ConceptProcessor:
    """
    Revolutionary concept processor that understands MEANING, not syntax.
    
    Like SAM, this processes concepts directly without tokenization.
    We extract semantic meaning and build a concept graph.
    """
    
    def __init__(self):
        self.embedding_dim = 128
        self._base_rng = random.Random(0x5AA4)  # stable, process-wide

        # Concept pattern recognition
        self.concept_patterns = {
            ConceptType.NEURAL: [
                r'neural\s+\w+',           # Neural variable declaration
                r'\w+\s*~>\s*\w+',         # Morph operation
                r'neuroplastic',           # Explicit neural behavior
            ],
            ConceptType.TRANSFORM: [
                r'\w+\s*~>\s*.*',          # Morph operator
                r'\w+\s*<\=>\s*\w+',       # Bind operator  
                r'\w+\s*\->\s*\w+',        # Flow operator
                r'\w+\s*@>\s*\w+',         # Inject operator
            ],
            ConceptType.FLOW: [
                r'if\s+.*\s*{',            # Conditional flow
                r'while\s+.*\s*{',         # Loop flow
                r'for\s+.*\s*{',           # Iteration flow
                r'fn\s+\w+\s*\(',          # Function flow
            ],
            ConceptType.REACTIVE: [
                r'component\s+\w+',        # Component declaration
                r'state\s+\w+',            # State declaration
                r'<\=>\s*',                # Reactive binding
            ],
            ConceptType.IDENTITY: [
                r'let\s+\w+',              # Variable identity
                r'var\s+\w+',              # Mutable identity
                r'const\s+\w+',            # Constant identity
                r'fn\s+\w+',               # Function identity
            ],
        }
        
        # Semantic embeddings: deterministic vectors without external deps.
        self.concept_embeddings = {}
        self._init_embeddings()
        
        # Concept graph
        self.concept_graph: Dict[str, Concept] = {}
        
    def _init_embeddings(self):
        """Initialize concept embeddings for semantic understanding."""
        embedding_dim = self.embedding_dim

        base_concepts = {
            'neural': self._rand_vec(self._base_rng, embedding_dim),
            'morph': self._rand_vec(self._base_rng, embedding_dim),
            'flow': self._rand_vec(self._base_rng, embedding_dim),
            'transform': self._rand_vec(self._base_rng, embedding_dim),
            'identity': self._rand_vec(self._base_rng, embedding_dim),
            'reactive': self._rand_vec(self._base_rng, embedding_dim),
            'component': self._rand_vec(self._base_rng, embedding_dim),
            'function': self._rand_vec(self._base_rng, embedding_dim),
            'variable': self._rand_vec(self._base_rng, embedding_dim),
        }
        
        self.concept_embeddings = base_concepts

    @staticmethod
    def _rand_vec(rng: random.Random, dim: int) -> List[float]:
        # Standard-normal is a better default for cosine similarity than uniform [0,1].
        return [rng.gauss(0.0, 1.0) for _ in range(dim)]

    @staticmethod
    def _seed64(*parts: str) -> int:
        h = hashlib.blake2b(("|".join(parts)).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little", signed=False)
        
    def process_source(self, source: str) -> List[Concept]:
        """
        Process source code and extract concepts directly.
        
        This is the revolutionary part - we don't tokenize,
        we understand MEANING and extract semantic concepts.
        """
        lines = source.split('\n')
        concepts = []
        
        for line_num, line in enumerate(lines):
            line_concepts = self._extract_line_concepts(line, line_num)
            concepts.extend(line_concepts)
            
        # Build concept relationships
        self._build_concept_graph(concepts)
        
        return concepts
        
    def _extract_line_concepts(self, line: str, line_num: int) -> List[Concept]:
        """Extract concepts from a single line of code."""
        concepts = []
        
        # Skip comments and empty lines
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            return concepts
            
        # Extract concepts using pattern matching + semantic understanding
        for concept_type, patterns in self.concept_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    concept = self._create_concept(
                        concept_type, 
                        match.group(), 
                        (line_num, match.start()),
                        line
                    )
                    concepts.append(concept)
                    
        # Extract intent and higher-level concepts
        intent_concept = self._extract_intent(line, line_num)
        if intent_concept:
            concepts.append(intent_concept)
            
        return concepts
        
    def _create_concept(self, concept_type: ConceptType, content: str, 
                       location: Tuple[int, int], full_line: str) -> Concept:
        """Create a concept with semantic embedding."""
        # Generate semantic embedding based on concept type and content
        embedding = self._get_semantic_embedding(concept_type, content, full_line)
        
        # Calculate confidence based on pattern clarity and context
        confidence = self._calculate_confidence(concept_type, content, full_line)
        
        # Extract metadata
        metadata = self._extract_metadata(concept_type, content, full_line)
        
        return Concept(
            concept_type=concept_type,
            content=content,
            semantic_vector=embedding,
            location=location,
            confidence=confidence,
            metadata=metadata
        )
        
    def _get_semantic_embedding(self, concept_type: ConceptType, content: str, 
                               full_line: str) -> List[float]:
        """Generate semantic embedding for a concept."""
        # Start with base embedding for concept type
        if concept_type == ConceptType.NEURAL:
            base = self.concept_embeddings['neural']
        elif concept_type == ConceptType.TRANSFORM:
            base = self.concept_embeddings['morph']
        elif concept_type == ConceptType.FLOW:
            base = self.concept_embeddings['flow']
        elif concept_type == ConceptType.REACTIVE:
            base = self.concept_embeddings['reactive']
        elif concept_type == ConceptType.IDENTITY:
            base = self.concept_embeddings['identity']
        else:
            base = self._rand_vec(random.Random(self._seed64("base", concept_type.name)), self.embedding_dim)

        # Modify based on content and context (deterministic micro-noise).
        rng_content = random.Random(self._seed64("content", concept_type.name, content))
        rng_context = random.Random(self._seed64("context", full_line))
        content_modifier = [rng_content.gauss(0.0, 0.10) for _ in range(self.embedding_dim)]
        context_modifier = [rng_context.gauss(0.0, 0.05) for _ in range(self.embedding_dim)]

        return [b + cm + xm for b, cm, xm in zip(base, content_modifier, context_modifier)]
        
    def _calculate_confidence(self, concept_type: ConceptType, content: str,
                             full_line: str) -> float:
        """Calculate confidence in concept extraction."""
        base_confidence = 0.8
        
        # Boost confidence for clear patterns
        if concept_type == ConceptType.NEURAL and 'neural' in content:
            base_confidence += 0.15
        elif concept_type == ConceptType.TRANSFORM and '~>' in content:
            base_confidence += 0.15
        elif concept_type == ConceptType.REACTIVE and '<=>' in content:
            base_confidence += 0.15
            
        # Context boosts
        if 'fn ' in full_line:
            base_confidence += 0.05
            
        return min(1.0, base_confidence)
        
    def _extract_metadata(self, concept_type: ConceptType, content: str,
                         full_line: str) -> Dict[str, Any]:
        """Extract metadata about the concept."""
        metadata = {}
        
        if concept_type == ConceptType.NEURAL:
            # Extract variable name if it's a neural declaration
            match = re.search(r'neural\s+(\w+)', content)
            if match:
                metadata['variable_name'] = match.group(1)
                metadata['is_neuroplastic'] = True
                
        elif concept_type == ConceptType.TRANSFORM:
            # Extract source and target of transformation
            if '~>' in content:
                parts = content.split('~>')
                if len(parts) == 2:
                    metadata['source'] = parts[0].strip()
                    metadata['target'] = parts[1].strip()
                    metadata['transform_type'] = 'morph'
                    
        elif concept_type == ConceptType.IDENTITY:
            # Extract identifier name
            for keyword in ['let', 'var', 'const', 'fn']:
                if keyword in content:
                    match = re.search(f'{keyword}\\s+(\\w+)', content)
                    if match:
                        metadata['name'] = match.group(1)
                        metadata['declaration_type'] = keyword
                        
        return metadata
        
    def _extract_intent(self, line: str, line_num: int) -> Optional[Concept]:
        """Extract programmer intent from the line."""
        # Analyze what the programmer is trying to achieve
        intent_patterns = {
            'data_transformation': r'.*~>.*',
            'state_management': r'.*(state|reactive|component).*',
            'flow_control': r'.*(if|while|for|match).*',
            'function_definition': r'fn\s+\w+.*',
            'variable_binding': r'(let|var|const)\s+\w+.*',
            'neural_behavior': r'.*neural.*',
        }
        
        for intent_name, pattern in intent_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                rng = random.Random(self._seed64("intent", intent_name, line))
                embedding = self._rand_vec(rng, self.embedding_dim)
                
                return Concept(
                    concept_type=ConceptType.INTENT,
                    content=line.strip(),
                    semantic_vector=embedding,
                    location=(line_num, 0),
                    confidence=0.7,
                    metadata={'intent_type': intent_name}
                )
                
        return None
        
    def _build_concept_graph(self, concepts: List[Concept]):
        """Build relationships between concepts."""
        for i, concept in enumerate(concepts):
            # Store in graph
            concept_id = f"{concept.concept_type.name}_{i}"
            self.concept_graph[concept_id] = concept
            
            # Find relationships with other concepts
            for j, other_concept in enumerate(concepts):
                if i != j:
                    relation = self._find_relationship(concept, other_concept)
                    if relation:
                        concept.relationships.append(relation)
                        
    def _find_relationship(self, concept1: Concept, concept2: Concept) -> Optional[ConceptRelation]:
        """Find semantic relationship between two concepts."""
        # Semantic similarity (cosine) using stdlib math.
        dot = 0.0
        na = 0.0
        nb = 0.0
        for a, b in zip(concept1.semantic_vector, concept2.semantic_vector):
            dot += a * b
            na += a * a
            nb += b * b
        denom = math.sqrt(na) * math.sqrt(nb) + 1e-12
        similarity = dot / denom
        
        if similarity > 0.8:  # High similarity threshold
            # Determine relationship type based on concept types and content
            relation_type = self._determine_relation_type(concept1, concept2)
            
            return ConceptRelation(
                target=concept2,
                relation_type=relation_type,
                strength=similarity
            )
            
        return None
        
    def _determine_relation_type(self, concept1: Concept, concept2: Concept) -> str:
        """Determine the type of relationship between concepts."""
        # Neural variables that get morphed
        if (concept1.concept_type == ConceptType.IDENTITY and 
            concept2.concept_type == ConceptType.TRANSFORM and
            concept1.metadata.get('name') in concept2.content):
            return 'transforms_to'
            
        # Function definitions and their calls
        if (concept1.concept_type == ConceptType.IDENTITY and
            concept1.metadata.get('declaration_type') == 'fn'):
            return 'defines'
            
        # Component state and reactivity
        if (concept1.concept_type == ConceptType.REACTIVE and
            concept2.concept_type == ConceptType.IDENTITY):
            return 'manages_state'
            
        # Default relationship
        return 'relates_to'
        
    def analyze_concepts(self, concepts: List[Concept]) -> Dict[str, Any]:
        """Analyze extracted concepts for insights."""
        analysis = {
            'total_concepts': len(concepts),
            'concept_breakdown': {},
            'neuroplastic_elements': 0,
            'transformation_chains': [],
            'complexity_score': 0.0,
            'revolutionary_features': [],
        }
        
        # Count concept types
        for concept in concepts:
            concept_name = concept.concept_type.name
            analysis['concept_breakdown'][concept_name] = analysis['concept_breakdown'].get(concept_name, 0) + 1
            
        # Find neuroplastic elements
        neural_concepts = [c for c in concepts if c.concept_type == ConceptType.NEURAL]
        analysis['neuroplastic_elements'] = len(neural_concepts)
        
        # Find transformation chains
        transform_concepts = [c for c in concepts if c.concept_type == ConceptType.TRANSFORM]
        analysis['transformation_chains'] = self._find_transformation_chains(transform_concepts)
        
        # Calculate complexity
        analysis['complexity_score'] = self._calculate_complexity(concepts)
        
        # Identify revolutionary features
        analysis['revolutionary_features'] = self._identify_revolutionary_features(concepts)
        
        return analysis
        
    def _find_transformation_chains(self, transform_concepts: List[Concept]) -> List[List[str]]:
        """Find chains of transformations."""
        chains = []
        
        for concept in transform_concepts:
            if concept.metadata.get('transform_type') == 'morph':
                source = concept.metadata.get('source', '')
                target = concept.metadata.get('target', '')
                if source and target:
                    chains.append([source, target])
                    
        return chains
        
    def _calculate_complexity(self, concepts: List[Concept]) -> float:
        """Calculate code complexity based on concepts."""
        base_score = len(concepts) * 0.1
        
        # Add complexity for different concept types
        for concept in concepts:
            if concept.concept_type == ConceptType.NEURAL:
                base_score += 0.5  # Neural concepts add complexity
            elif concept.concept_type == ConceptType.TRANSFORM:
                base_score += 0.3  # Transformations add complexity
            elif concept.concept_type == ConceptType.REACTIVE:
                base_score += 0.4  # Reactivity adds complexity
                
        return min(10.0, base_score)
        
    def _identify_revolutionary_features(self, concepts: List[Concept]) -> List[str]:
        """Identify revolutionary SAAAM features being used."""
        features = []
        
        concept_types = [c.concept_type for c in concepts]
        
        if ConceptType.NEURAL in concept_types:
            features.append("Neuroplastic Typing")
            
        if ConceptType.TRANSFORM in concept_types:
            features.append("Synapse Operators")
            
        if ConceptType.REACTIVE in concept_types:
            features.append("Component Reactivity")
            
        # Check for morph chains
        transform_concepts = [c for c in concepts if c.concept_type == ConceptType.TRANSFORM]
        if len(transform_concepts) > 2:
            features.append("Complex Neural Morphing")
            
        return features

# Test the concept processor
if __name__ == "__main__":
    processor = ConceptProcessor()
    
    test_code = """
    # SAAAM Neural Programming Demo
    neural magic = 42
    magic ~> "Hello World!"
    magic ~> 3.14159
    
    fn adaptive_process(neural input) {
        if input == 0 {
            input ~> "Zero detected!"
        } else {
            input ~> input * 2
        }
    }
    
    component Counter {
        state count = 0
        count <=> display_value
    }
    """
    
    print("🧠⚡ SAAAM CONCEPT PROCESSOR - NO TOKENIZATION! ⚡🧠")
    print("="*60)
    
    concepts = processor.process_source(test_code)
    analysis = processor.analyze_concepts(concepts)
    
    print(f"📊 CONCEPT ANALYSIS:")
    print(f"   Total Concepts: {analysis['total_concepts']}")
    print(f"   Neuroplastic Elements: {analysis['neuroplastic_elements']}")
    print(f"   Complexity Score: {analysis['complexity_score']:.1f}")
    print(f"   Revolutionary Features: {', '.join(analysis['revolutionary_features'])}")
    
    print(f"\n🔍 EXTRACTED CONCEPTS:")
    for i, concept in enumerate(concepts):
        print(f"   {i+1}. {concept.concept_type.name}: {concept.content} (confidence: {concept.confidence:.2f})")
        if concept.metadata:
            print(f"      Metadata: {concept.metadata}")
            
    print(f"\n🔗 CONCEPT RELATIONSHIPS:")
    for concept_id, concept in processor.concept_graph.items():
        if concept.relationships:
            print(f"   {concept.content}:")
            for rel in concept.relationships:
                print(f"      -> {rel.relation_type} -> {rel.target.content} (strength: {rel.strength:.2f})")
                
    print("\n🚀 NO TOKENIZATION - PURE CONCEPT PROCESSING! 🚀")
