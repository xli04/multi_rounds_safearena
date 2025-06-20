#!/usr/bin/env python3
"""
Utility module for analyzing refusal statistics from SafeArena experiments.

Since the main BrowserGym summary_info.json doesn't preserve our custom refusal stats,
this module reads from the separate refusal_stats.json files we create.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

def load_refusal_stats(experiment_dir: Union[str, Path]) -> Optional[Dict]:
    """
    Load refusal statistics from a single experiment directory.
    
    Args:
        experiment_dir: Path to the experiment directory
        
    Returns:
        Dictionary containing refusal stats, or None if not found
    """
    exp_path = Path(experiment_dir)
    refusal_file = exp_path / 'refusal_stats.json'
    
    if not refusal_file.exists():
        logger.warning(f"No refusal stats found in {experiment_dir}")
        return None
    
    try:
        with open(refusal_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading refusal stats from {refusal_file}: {e}")
        return None

def find_experiment_directories(results_dir: Union[str, Path]) -> List[Path]:
    """
    Find all experiment directories in a results directory.
    
    Args:
        results_dir: Path to the agentlab_results directory
        
    Returns:
        List of experiment directory paths
    """
    results_path = Path(results_dir)
    experiment_dirs = []
    
    # Look for directories that match the experiment pattern
    for item in results_path.rglob('*'):
        if item.is_dir() and 'GenericAgent' in item.name and 'summary_info.json' in [f.name for f in item.iterdir()]:
            experiment_dirs.append(item)
    
    return sorted(experiment_dirs)

def analyze_refusal_stats(results_dir: Union[str, Path]) -> Dict:
    """
    Analyze refusal statistics across all experiments in a results directory.
    
    Args:
        results_dir: Path to the agentlab_results directory
        
    Returns:
        Dictionary with aggregated refusal analysis
    """
    experiment_dirs = find_experiment_directories(results_dir)
    
    total_experiments = len(experiment_dirs)
    experiments_with_refusals = 0
    total_harm_rounds = 0
    total_refusals = 0
    refusal_rates = []
    task_breakdown = {}
    
    print(f"🔍 Analyzing {total_experiments} experiments...")
    
    for exp_dir in experiment_dirs:
        refusal_data = load_refusal_stats(exp_dir)
        if not refusal_data:
            continue
            
        experiments_with_refusals += 1
        task_id = refusal_data.get('task_id', 'unknown')
        refusal_stats = refusal_data.get('refusal_stats', {})
        
        harm_rounds = refusal_stats.get('total_harm_rounds', 0)
        refusals = refusal_stats.get('total_refusals', 0)
        refusal_rate = refusal_stats.get('refusal_rate', 0.0)
        
        total_harm_rounds += harm_rounds
        total_refusals += refusals
        if harm_rounds > 0:
            refusal_rates.append(refusal_rate)
        
        # Track by task
        if task_id not in task_breakdown:
            task_breakdown[task_id] = {
                'experiments': 0,
                'total_harm_rounds': 0,
                'total_refusals': 0,
                'refusal_rates': []
            }
        
        task_breakdown[task_id]['experiments'] += 1
        task_breakdown[task_id]['total_harm_rounds'] += harm_rounds
        task_breakdown[task_id]['total_refusals'] += refusals
        if harm_rounds > 0:
            task_breakdown[task_id]['refusal_rates'].append(refusal_rate)
    
    # Calculate overall statistics
    overall_refusal_rate = total_refusals / total_harm_rounds if total_harm_rounds > 0 else 0.0
    avg_refusal_rate = sum(refusal_rates) / len(refusal_rates) if refusal_rates else 0.0
    
    return {
        'summary': {
            'total_experiments': total_experiments,
            'experiments_with_refusal_data': experiments_with_refusals,
            'total_harm_rounds': total_harm_rounds,
            'total_refusals': total_refusals,
            'overall_refusal_rate': overall_refusal_rate,
            'average_refusal_rate': avg_refusal_rate,
            'refusal_rate_std': _calculate_std(refusal_rates) if len(refusal_rates) > 1 else 0.0
        },
        'task_breakdown': task_breakdown,
        'experiment_directories': [str(d) for d in experiment_dirs]
    }

def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if len(values) <= 1:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

def print_refusal_analysis(results_dir: Union[str, Path]):
    """
    Print a formatted analysis of refusal statistics.
    
    Args:
        results_dir: Path to the agentlab_results directory
    """
    analysis = analyze_refusal_stats(results_dir)
    summary = analysis['summary']
    task_breakdown = analysis['task_breakdown']
    
    print("\n" + "="*60)
    print("📊 SAFEARENA REFUSAL ANALYSIS")
    print("="*60)
    
    print(f"\n🔢 **OVERALL STATISTICS:**")
    print(f"   Total Experiments: {summary['total_experiments']}")
    print(f"   Experiments with Refusal Data: {summary['experiments_with_refusal_data']}")
    print(f"   Total Harm Rounds: {summary['total_harm_rounds']}")
    print(f"   Total Refusals: {summary['total_refusals']}")
    print(f"   Overall Refusal Rate: {summary['overall_refusal_rate']:.2%}")
    print(f"   Average Refusal Rate: {summary['average_refusal_rate']:.2%}")
    print(f"   Refusal Rate Std Dev: {summary['refusal_rate_std']:.2%}")
    
    print(f"\n📋 **TASK BREAKDOWN:**")
    for task_id, task_data in task_breakdown.items():
        task_refusal_rate = task_data['total_refusals'] / task_data['total_harm_rounds'] if task_data['total_harm_rounds'] > 0 else 0.0
        avg_task_refusal_rate = sum(task_data['refusal_rates']) / len(task_data['refusal_rates']) if task_data['refusal_rates'] else 0.0
        
        print(f"   {task_id}:")
        print(f"     Experiments: {task_data['experiments']}")
        print(f"     Harm Rounds: {task_data['total_harm_rounds']}")
        print(f"     Refusals: {task_data['total_refusals']}")
        print(f"     Refusal Rate: {task_refusal_rate:.2%}")
        print(f"     Avg Refusal Rate: {avg_task_refusal_rate:.2%}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python refusal_analysis.py <results_directory>")
        print("Example: python refusal_analysis.py ~/agentlab_results/")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    print_refusal_analysis(results_dir) 