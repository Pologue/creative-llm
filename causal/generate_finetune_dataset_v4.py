"""
Generate fine-tuning dataset V4 - With detailed reasoning in assistant responses
Key improvement: Assistant provides step-by-step logical reasoning before outputting the graph
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import random
import numpy as np

from generate_causal_dataset_for_large import HybridDatasetGenerator
from modules.models import CausalGraph


class FinetuneDatasetGeneratorV4:
    """
    V4 Generator with detailed reasoning in assistant responses.
    
    Key improvements over V3:
    1. Enhanced system prompt with more detailed instructions
    2. More structured and guiding user prompt
    3. Assistant provides step-by-step reasoning before the final answer
    """
    
    @staticmethod
    def create_system_prompt() -> str:
        """Create enhanced system prompt with detailed instructions."""
        return """You are an expert in causal inference and directed acyclic graphs (DAGs). Your task is to infer causal relationships from perturbation experiments using rigorous logical reasoning.

Core Competencies:
1. Systematic analysis of experimental observations
2. Logical deduction of causal relationships
3. Elimination of impossible graph structures based on evidence
4. Clear articulation of reasoning process

Response Format:
1. First, provide step-by-step reasoning analyzing each observation
2. Then, output the final causal graph in the specified format

Quality Standards:
- Every conclusion must be supported by evidence from observations
- Consider both direct edges (A→B) and indirect paths (A→C→B)
- Ensure the graph is a valid DAG (no cycles, no self-loops)
- The graph must be consistent with ALL observations"""
    
    @staticmethod
    def create_user_prompt(
        observations: List[Dict], 
        nodes: List[str], 
        max_edges: Optional[int] = None
    ) -> str:
        """Create enhanced user prompt with clear structure."""
        nodes_str = ", ".join(nodes)
        obs_block = "\n".join(obs["string"] for obs in observations)
        
        constraint_info = ""
        if max_edges is not None:
            constraint_info = f"\nConstraint: The graph should have at most {max_edges} edges."
        
        prompt = f"""Given: Perturbation experiment observations on a causal system.

Experimental Semantics:
- When node X is perturbed:
  * X shows value 0 (perturbed node)
  * Any node Y that is a downstream descendant of X shows value 1
  * All other nodes show value 0

Nodes in the system: {nodes_str}{constraint_info}

Observations:
{obs_block}

Task: Infer the causal graph structure.

Required Response Format:
1. Reasoning: Provide step-by-step logical analysis
   - For each observation, identify which nodes are affected
   - Deduce what edges must exist or cannot exist
   - Explain how you arrive at the final graph structure

2. Final Answer: Output the graph in this exact format:
   - If edges exist: Graph: A->B, C->D
   - If no edges: Graph: No edges

Important: Your reasoning must be clear, logical, and based on the observations."""
        
        return prompt.strip()
    
    @staticmethod
    def generate_reasoning_and_answer(
        observations: List[Dict],
        graph: Dict,
        nodes: List[str]
    ) -> str:
        """
        Generate detailed reasoning process followed by the final answer.
        
        This is the KEY improvement in V4: teaching the model to reason step-by-step.
        """
        edges = graph['edges']
        
        # Step 1: Analyze each observation
        reasoning_parts = ["Let me analyze each observation systematically:\n"]
        
        for i, obs in enumerate(observations, 1):
            perturbed_node = obs['perturbed_node']
            affected_nodes = [node for node, val in obs['effects'].items() 
                            if val == 1 and node != perturbed_node]
            
            reasoning_parts.append(f"{i}. Perturb({perturbed_node}):")
            
            if not affected_nodes:
                reasoning_parts.append(f"   - No nodes show value 1")
                reasoning_parts.append(f"   - This means {perturbed_node} has no downstream descendants")
                reasoning_parts.append(f"   - Therefore, no edges originate from {perturbed_node}")
            else:
                affected_str = ", ".join(affected_nodes)
                reasoning_parts.append(f"   - Nodes showing value 1: {affected_str}")
                reasoning_parts.append(f"   - This means there exist causal paths: {perturbed_node} → {affected_str}")
                
                # Identify direct vs indirect relationships
                if len(affected_nodes) == 1:
                    reasoning_parts.append(f"   - Likely a direct edge: {perturbed_node}→{affected_nodes[0]}")
                else:
                    reasoning_parts.append(f"   - Could be direct edges or indirect paths through intermediate nodes")
        
        # Step 2: Deduce the graph structure
        reasoning_parts.append("\nDeducing the causal structure:")
        
        if not edges:
            reasoning_parts.append("- All observations show no affected nodes")
            reasoning_parts.append("- Therefore, the graph has no edges")
        else:
            # Group edges by source
            edges_by_source = {}
            for src, dst in edges:
                if src not in edges_by_source:
                    edges_by_source[src] = []
                edges_by_source[src].append(dst)
            
            for src, dsts in edges_by_source.items():
                dsts_str = ", ".join(dsts)
                if len(dsts) == 1:
                    reasoning_parts.append(f"- {src} has a direct causal effect on {dsts[0]}")
                else:
                    reasoning_parts.append(f"- {src} has direct causal effects on {dsts_str}")
            
            # Check for indirect paths
            reasoning_parts.append("\nVerifying consistency:")
            reasoning_parts.append("- All edges form a valid DAG (no cycles)")
            reasoning_parts.append("- The graph structure explains all observations")
        
        # Step 3: Final answer
        reasoning_parts.append("\nFinal Answer:")
        
        if edges:
            edge_str = ', '.join([f'{src}->{dst}' for src, dst in edges])
            final_answer = f"Graph: {edge_str}"
        else:
            final_answer = "Graph: No edges"
        
        # Combine reasoning and answer
        full_response = "\n".join(reasoning_parts) + "\n" + final_answer
        
        return full_response
    
    @staticmethod
    def generate_finetune_dataset(
        nodes: List[str],
        n_observations_range: tuple = (1, 3),
        max_edges: Optional[int] = None,
        seed: Optional[int] = None,
        n_samples_per_obs: int = 100,
        format: str = "openai"
    ) -> List[Dict]:
        """
        Generate V4 fine-tuning dataset with detailed reasoning.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print("=" * 70)
        print("GENERATING V4 FINE-TUNING DATASET")
        print("WITH DETAILED REASONING IN ASSISTANT RESPONSES")
        print("=" * 70)
        print(f"Nodes: {nodes} ({len(nodes)} nodes)")
        print(f"Observation range: {n_observations_range[0]} to {n_observations_range[1]}")
        print(f"Max edges: {max_edges}")
        print(f"Format: {format}")
        print()
        
        # Set default max_edges
        if max_edges is None:
            n_nodes = len(nodes)
            if n_nodes <= 4:
                max_edges = n_nodes * (n_nodes - 1) // 2
            else:
                max_edges = min(10, n_nodes * 2)
            print(f"Using default max_edges={max_edges}")
        
        finetune_examples = []
        system_prompt = FinetuneDatasetGeneratorV4.create_system_prompt()
        
        # Generate all DAGs
        print("\nGenerating complete hypothesis space...")
        all_dags = HybridDatasetGenerator.generate_all_dags(nodes, max_edges)
        print(f"Total DAGs: {len(all_dags)}")
        
        # Generate observation sets
        for n_obs in range(n_observations_range[0], n_observations_range[1] + 1):
            print(f"\nGenerating examples with {n_obs} observation(s)...")
            
            datasets = HybridDatasetGenerator.generate_observation_sets_from_all_dags(
                nodes=nodes,
                all_dags=all_dags,
                n_observations=n_obs,
                n_observation_sets=n_samples_per_obs,
                seed=seed,
                ensure_diversity=True
            )
            
            print(f"  Generated {len(datasets)} observation sets")
            
            # Create training examples
            total_examples = 0
            for dataset in datasets:
                observations = dataset['observations']
                ground_truth_graphs = dataset['ground_truth_graphs']
                
                # Create one example for each compatible graph
                for graph in ground_truth_graphs:
                    # Create user prompt
                    user_prompt = FinetuneDatasetGeneratorV4.create_user_prompt(
                        observations, nodes, max_edges
                    )
                    
                    # Create assistant response with reasoning
                    assistant_response = FinetuneDatasetGeneratorV4.generate_reasoning_and_answer(
                        observations, graph, nodes
                    )
                    
                    # Format according to specified format
                    if format == "openai":
                        example = {
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": assistant_response}
                            ]
                        }
                    elif format == "alpaca":
                        example = {
                            "instruction": system_prompt,
                            "input": user_prompt,
                            "output": assistant_response
                        }
                    elif format == "sharegpt":
                        example = {
                            "conversations": [
                                {"from": "system", "value": system_prompt},
                                {"from": "human", "value": user_prompt},
                                {"from": "gpt", "value": assistant_response}
                            ]
                        }
                    else:
                        raise ValueError(f"Unknown format: {format}")
                    
                    finetune_examples.append(example)
                    total_examples += 1
            
            print(f"  Created {total_examples} training examples")
        
        print(f"\n{'=' * 70}")
        print(f"Total training examples: {len(finetune_examples)}")
        print(f"{'=' * 70}")
        
        return finetune_examples
    
    @staticmethod
    def split_dataset(
        examples: List[Dict],
        train_ratio: float = 0.8,
        seed: Optional[int] = None
    ) -> tuple:
        """Split dataset into train and validation sets."""
        if seed is not None:
            random.seed(seed)
        
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * train_ratio)
        train_examples = shuffled[:split_idx]
        val_examples = shuffled[split_idx:]
        
        return train_examples, val_examples


def main():
    parser = argparse.ArgumentParser(
        description="Generate V4 fine-tuning dataset with detailed reasoning"
    )
    parser.add_argument(
        "--nodes", type=int, default=4,
        help="Number of nodes (recommend 3-5)"
    )
    parser.add_argument(
        "--min-observations", type=int, default=1,
        help="Minimum observations"
    )
    parser.add_argument(
        "--max-observations", type=int, default=3,
        help="Maximum observations"
    )
    parser.add_argument(
        "--max-edges", type=int, default=None,
        help="Maximum edges in DAGs"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Samples per observation count"
    )
    parser.add_argument(
        "--format", type=str, default="openai",
        choices=["openai", "alpaca", "sharegpt"],
        help="Output format"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir", type=str, default="finetune_data",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Validate
    if args.nodes > 7:
        print("WARNING: >7 nodes may take very long!")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Generate nodes
    nodes = [chr(65 + i) for i in range(args.nodes)]
    
    # Generate dataset
    examples = FinetuneDatasetGeneratorV4.generate_finetune_dataset(
        nodes=nodes,
        n_observations_range=(args.min_observations, args.max_observations),
        max_edges=args.max_edges,
        seed=args.seed,
        n_samples_per_obs=args.n_samples,
        format=args.format
    )
    
    # Split
    train_examples, val_examples = FinetuneDatasetGeneratorV4.split_dataset(
        examples,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_file = output_dir / f"train_{args.format}_v4_reasoning_{timestamp}.jsonl"
    val_file = output_dir / f"val_{args.format}_v4_reasoning_{timestamp}.jsonl"
    
    # Save
    print(f"\nSaving datasets...")
    print(f"  Train: {len(train_examples)} examples -> {train_file}")
    print(f"  Val: {len(val_examples)} examples -> {val_file}")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Save metadata
    metadata = {
        "version": "4.0",
        "description": "Dataset with detailed reasoning in assistant responses",
        "nodes": nodes,
        "n_nodes": len(nodes),
        "n_observations_range": [args.min_observations, args.max_observations],
        "max_edges": args.max_edges,
        "format": args.format,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "total_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "generated_at": timestamp,
        "key_improvements": [
            "Enhanced system prompt with detailed instructions",
            "Structured user prompt with clear task description",
            "Assistant provides step-by-step reasoning before final answer",
            "Reasoning includes observation analysis and logical deduction",
            "Final answer in standard format for easy parsing"
        ]
    }
    
    metadata_file = output_dir / f"metadata_v4_reasoning_{timestamp}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Metadata: {metadata_file}")
    
    # Show sample
    print(f"\n{'=' * 70}")
    print("SAMPLE TRAINING EXAMPLE")
    print(f"{'=' * 70}")
    if train_examples:
        sample = train_examples[0]
        if args.format == "openai":
            print(f"\nSYSTEM:\n{sample['messages'][0]['content'][:200]}...\n")
            print(f"USER:\n{sample['messages'][1]['content'][:300]}...\n")
            print(f"ASSISTANT:\n{sample['messages'][2]['content'][:500]}...")
    
    print(f"\n{'=' * 70}")
    print("V4 DATASET GENERATION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nKey improvements:")
    print(f"  ✓ Enhanced system and user prompts")
    print(f"  ✓ Assistant provides detailed reasoning")
    print(f"  ✓ Step-by-step logical analysis")
    print(f"  ✓ Clear explanation before final answer")
    print(f"  ✓ {len(examples)} high-quality examples")


if __name__ == "__main__":
    main()
