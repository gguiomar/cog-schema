import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from tqdm import tqdm

from observers.agents.bayes_agent import BayesAgent
from observers.agents.map_agent import MAPAgent


class LLMBayesianComparison:
    """
    Compare LLM performance with optimal Bayesian observer on bias detection tasks.
    """
    
    def __init__(self, task_parameters: Dict = None):
        """
        Initialize the comparison framework.
        
        Parameters:
        -----------
        task_parameters : Dict
            Task parameters including k, p_t, p_f for the Bayesian agent
        """
        # Default task parameters for bias detection task
        self.task_params = task_parameters or {
            'k': 4,  # 4 quadrants
            'p_t': 0.9,  # probability of RED when cue matches biased quadrant
            'p_f': 0.5   # probability of RED when cue doesn't match biased quadrant
        }
        
        # Initialize agents
        self.bayes_agent = BayesAgent(
            k=self.task_params['k'],
            p_t=self.task_params['p_t'],
            p_f=self.task_params['p_f']
        )
        self.map_agent = MAPAgent(k=self.task_params['k'])
        
    def extract_llm_data_from_log(self, log_file_path: str) -> Dict:
        """
        Extract LLM trial data from TaskManager log files.
        
        Parameters:
        -----------
        log_file_path : str
            Path to the JSON log file
            
        Returns:
        --------
        Dict
            Extracted and formatted LLM data
        """
        with open(log_file_path, 'r') as f:
            data = json.load(f)
        
        extracted_data = {
            'agent_name': data['metrics']['agent'],
            'task_config': {
                'n_rounds': data['metrics']['n_rounds'],
                'n_quadrants': data['metrics']['n_quadrants'],
                'n_trials': data['metrics']['n_trials']
            },
            'trials': []
        }
        
        # Process each simulation
        for sim in data['raw_results']:
            for trial in sim['trials']:
                trial_data = {
                    'rounds': [],
                    'final_choice': trial['final_choice'],
                    'correct_quadrant': trial['correct_quadrant'],
                    'success': trial['success']
                }
                
                # Process each round
                for round_data in trial['rounds']:
                    # Parse available cues
                    available_cues = round_data['available_cues'].split(', ')
                    chosen_cue = round_data['choice']
                    result_color = round_data['result']
                    
                    # Convert quadrant number to 0-indexed
                    quadrant = round_data['quadrant'] - 1 if round_data['quadrant'] else None
                    
                    trial_data['rounds'].append({
                        'available_cues': available_cues,
                        'chosen_cue': chosen_cue,
                        'result_color': result_color,
                        'quadrant': quadrant,
                        'round_time': round_data.get('round_time', 0)
                    })
                
                extracted_data['trials'].append(trial_data)
        
        return extracted_data
    
    def convert_to_bayesian_format(self, llm_trial_data: Dict) -> List[Tuple[int, int]]:
        """
        Convert LLM trial data to format suitable for Bayesian agent.
        
        Parameters:
        -----------
        llm_trial_data : Dict
            Single trial data from LLM
            
        Returns:
        --------
        List[Tuple[int, int]]
            List of (cue_location, color) observations
        """
        observations = []
        
        for round_data in llm_trial_data['rounds']:
            chosen_cue = round_data['chosen_cue']
            result_color = round_data['result_color']
            
            # Convert cue letter to index (A=0, B=1, C=2, D=3)
            cue_idx = ord(chosen_cue.upper()) - ord('A')
            
            # Convert color to binary (RED=1, GREEN=0)
            color_binary = 1 if result_color.upper() == 'RED' else 0
            
            observations.append((cue_idx, color_binary))
        
        return observations
    
    def run_optimal_bayesian_observer(self, observations: List[Tuple[int, int]], 
                                    true_target: int) -> Dict:
        """
        Run the optimal Bayesian observer on the same sequence of observations.
        
        Parameters:
        -----------
        observations : List[Tuple[int, int]]
            List of (cue_location, color) observations
        true_target : int
            The true biased quadrant (0-indexed)
            
        Returns:
        --------
        Dict
            Bayesian observer results including posterior evolution and decision
        """
        # Reset the Bayesian agent
        self.bayes_agent.reset()
        
        # Track posterior evolution
        posterior_evolution = [self.bayes_agent.posterior.copy()]
        surprise_metrics = []
        
        # Process each observation
        for cue, color in observations:
            shannon_surprise, bayesian_surprise, pred_prob = self.bayes_agent.update(cue, color)
            
            posterior_evolution.append(self.bayes_agent.posterior.copy())
            surprise_metrics.append({
                'shannon_surprise': shannon_surprise,
                'bayesian_surprise': bayesian_surprise,
                'predictive_probability': pred_prob
            })
        
        # Get final decision using MAP
        optimal_decision = self.map_agent.get_decision(self.bayes_agent.posterior)
        map_confidence = self.map_agent.get_map_probability(self.bayes_agent.posterior)
        optimal_correct = self.map_agent.is_correct(optimal_decision, true_target)
        
        return {
            'optimal_decision': optimal_decision,
            'optimal_correct': optimal_correct,
            'map_confidence': map_confidence,
            'final_posterior': self.bayes_agent.posterior.copy(),
            'posterior_evolution': posterior_evolution,
            'surprise_metrics': surprise_metrics,
            'final_entropy': self.bayes_agent.entropy
        }
    
    def compare_single_trial(self, llm_trial_data: Dict) -> Dict:
        """
        Compare LLM performance with optimal Bayesian observer for a single trial.
        
        Parameters:
        -----------
        llm_trial_data : Dict
            Single trial data from LLM
            
        Returns:
        --------
        Dict
            Comparison results for this trial
        """
        # Extract LLM decision and correctness
        llm_choice = llm_trial_data['final_choice']
        llm_correct = llm_trial_data['success']
        
        # Convert correct quadrant to 0-indexed
        correct_quadrant_str = str(llm_trial_data['correct_quadrant'])
        if correct_quadrant_str.isdigit():
            true_target = int(correct_quadrant_str)
        else:
            # If it's already a letter, convert to index
            true_target = ord(correct_quadrant_str.upper()) - ord('A')
        
        # Convert LLM observations to Bayesian format
        observations = self.convert_to_bayesian_format(llm_trial_data)
        
        # Run optimal Bayesian observer
        bayesian_results = self.run_optimal_bayesian_observer(observations, true_target)
        
        # Convert optimal decision back to letter
        optimal_choice_letter = chr(ord('A') + bayesian_results['optimal_decision'])
        
        # Calculate agreement
        agreement = (llm_choice.upper() == optimal_choice_letter)
        
        return {
            'llm_choice': llm_choice,
            'llm_correct': llm_correct,
            'optimal_choice': optimal_choice_letter,
            'optimal_correct': bayesian_results['optimal_correct'],
            'agreement': agreement,
            'true_target': chr(ord('A') + true_target),
            'map_confidence': bayesian_results['map_confidence'],
            'final_posterior': bayesian_results['final_posterior'],
            'posterior_evolution': bayesian_results['posterior_evolution'],
            'surprise_metrics': bayesian_results['surprise_metrics'],
            'final_entropy': bayesian_results['final_entropy'],
            'observations': observations
        }
    
    def compare_llm_to_optimal_bayesian_observer(self, llm_data: Union[str, Dict], 
                                               return_detailed_analysis: bool = True) -> Dict:
        """
        Main function to compare LLM performance with optimal Bayesian observer.
        
        Parameters:
        -----------
        llm_data : Union[str, Dict]
            Either path to log file or extracted LLM data
        return_detailed_analysis : bool
            Whether to return detailed trial-by-trial analysis
            
        Returns:
        --------
        Dict
            Comprehensive comparison results
        """
        # Load data if path is provided
        if isinstance(llm_data, str):
            llm_data = self.extract_llm_data_from_log(llm_data)
        
        # Process all trials
        trial_results = []
        
        print(f"Analyzing {len(llm_data['trials'])} trials...")
        for i, trial in enumerate(tqdm(llm_data['trials'])):
            trial_result = self.compare_single_trial(trial)
            trial_result['trial_id'] = i
            trial_results.append(trial_result)
        
        # Calculate summary metrics
        llm_accuracy = np.mean([t['llm_correct'] for t in trial_results])
        optimal_accuracy = np.mean([t['optimal_correct'] for t in trial_results])
        agreement_rate = np.mean([t['agreement'] for t in trial_results])
        performance_gap = optimal_accuracy - llm_accuracy
        
        # Calculate confidence statistics
        map_confidences = [t['map_confidence'] for t in trial_results]
        avg_confidence = np.mean(map_confidences)
        
        # Analyze disagreement patterns
        disagreements = [t for t in trial_results if not t['agreement']]
        disagreement_analysis = self._analyze_disagreement_patterns(disagreements)
        
        # Compile results
        results = {
            'summary_metrics': {
                'agent_name': llm_data['agent_name'],
                'task_config': llm_data['task_config'],
                'n_trials_analyzed': len(trial_results),
                'llm_accuracy': llm_accuracy,
                'optimal_accuracy': optimal_accuracy,
                'agreement_rate': agreement_rate,
                'performance_gap': performance_gap,
                'avg_map_confidence': avg_confidence,
                'std_map_confidence': np.std(map_confidences)
            },
            'disagreement_analysis': disagreement_analysis
        }
        
        if return_detailed_analysis:
            results['trial_by_trial_analysis'] = trial_results
        
        return results
    
    def _analyze_disagreement_patterns(self, disagreements: List[Dict]) -> Dict:
        """
        Analyze patterns in cases where LLM and optimal observer disagree.
        
        Parameters:
        -----------
        disagreements : List[Dict]
            List of trials where LLM and optimal observer disagreed
            
        Returns:
        --------
        Dict
            Analysis of disagreement patterns
        """
        if not disagreements:
            return {'n_disagreements': 0}
        
        # Count outcomes when they disagree
        llm_correct_when_disagree = sum(1 for d in disagreements if d['llm_correct'])
        optimal_correct_when_disagree = sum(1 for d in disagreements if d['optimal_correct'])
        
        # Analyze confidence in disagreements
        confidences_when_disagree = [d['map_confidence'] for d in disagreements]
        
        # Analyze choice patterns
        llm_choices = [d['llm_choice'] for d in disagreements]
        optimal_choices = [d['optimal_choice'] for d in disagreements]
        
        from collections import Counter
        llm_choice_dist = Counter(llm_choices)
        optimal_choice_dist = Counter(optimal_choices)
        
        return {
            'n_disagreements': len(disagreements),
            'disagreement_rate': len(disagreements) / (len(disagreements) + 1),  # Avoid division by zero
            'llm_correct_when_disagree': llm_correct_when_disagree,
            'optimal_correct_when_disagree': optimal_correct_when_disagree,
            'avg_confidence_when_disagree': np.mean(confidences_when_disagree),
            'llm_choice_distribution': dict(llm_choice_dist),
            'optimal_choice_distribution': dict(optimal_choice_dist)
        }
    
    def plot_comparison_results(self, results: Dict, save_path: str = None) -> None:
        """
        Create visualization of the comparison results.
        
        Parameters:
        -----------
        results : Dict
            Results from compare_llm_to_optimal_bayesian_observer
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract data for plotting
        if 'trial_by_trial_analysis' not in results:
            print("Detailed analysis not available for plotting")
            return
        
        trials = results['trial_by_trial_analysis']
        summary = results['summary_metrics']
        
        # Plot 1: Accuracy comparison
        ax = axes[0, 0]
        accuracies = [summary['llm_accuracy'], summary['optimal_accuracy']]
        labels = ['LLM', 'Optimal Bayesian']
        colors = ['skyblue', 'orange']
        bars = ax.bar(labels, accuracies, color=colors)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # Plot 2: Agreement rate
        ax = axes[0, 1]
        agreement_rate = summary['agreement_rate']
        ax.bar(['Agreement Rate'], [agreement_rate], color='lightgreen')
        ax.set_ylabel('Rate')
        ax.set_title('LLM-Optimal Agreement Rate')
        ax.set_ylim(0, 1)
        ax.text(0, agreement_rate + 0.01, f'{agreement_rate:.3f}', 
               ha='center', va='bottom')
        
        # Plot 3: Performance gap
        ax = axes[0, 2]
        performance_gap = summary['performance_gap']
        color = 'red' if performance_gap > 0 else 'green'
        ax.bar(['Performance Gap'], [performance_gap], color=color, alpha=0.7)
        ax.set_ylabel('Gap (Optimal - LLM)')
        ax.set_title('Performance Gap')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.text(0, performance_gap + 0.005 if performance_gap > 0 else performance_gap - 0.005,
               f'{performance_gap:.3f}', ha='center', 
               va='bottom' if performance_gap > 0 else 'top')
        
        # Plot 4: MAP confidence distribution
        ax = axes[1, 0]
        confidences = [t['map_confidence'] for t in trials]
        ax.hist(confidences, bins=20, alpha=0.7, color='purple')
        ax.set_xlabel('MAP Confidence')
        ax.set_ylabel('Frequency')
        ax.set_title('MAP Confidence Distribution')
        ax.axvline(x=np.mean(confidences), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(confidences):.3f}')
        ax.legend()
        
        # Plot 5: Choice distribution comparison
        ax = axes[1, 1]
        llm_choices = [t['llm_choice'] for t in trials]
        optimal_choices = [t['optimal_choice'] for t in trials]
        
        from collections import Counter
        llm_dist = Counter(llm_choices)
        optimal_dist = Counter(optimal_choices)
        
        choices = ['A', 'B', 'C', 'D']
        llm_counts = [llm_dist.get(c, 0) for c in choices]
        optimal_counts = [optimal_dist.get(c, 0) for c in choices]
        
        x = np.arange(len(choices))
        width = 0.35
        
        ax.bar(x - width/2, llm_counts, width, label='LLM', alpha=0.7)
        ax.bar(x + width/2, optimal_counts, width, label='Optimal', alpha=0.7)
        ax.set_xlabel('Choice')
        ax.set_ylabel('Count')
        ax.set_title('Choice Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(choices)
        ax.legend()
        
        # Plot 6: Accuracy by confidence quartiles
        ax = axes[1, 2]
        # Divide trials into confidence quartiles
        sorted_trials = sorted(trials, key=lambda x: x['map_confidence'])
        n_trials = len(sorted_trials)
        quartile_size = n_trials // 4
        
        quartile_accuracies_llm = []
        quartile_accuracies_opt = []
        quartile_labels = []
        
        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else n_trials
            quartile_trials = sorted_trials[start_idx:end_idx]
            
            llm_acc = np.mean([t['llm_correct'] for t in quartile_trials])
            opt_acc = np.mean([t['optimal_correct'] for t in quartile_trials])
            
            quartile_accuracies_llm.append(llm_acc)
            quartile_accuracies_opt.append(opt_acc)
            quartile_labels.append(f'Q{i+1}')
        
        x = np.arange(len(quartile_labels))
        ax.bar(x - width/2, quartile_accuracies_llm, width, label='LLM', alpha=0.7)
        ax.bar(x + width/2, quartile_accuracies_opt, width, label='Optimal', alpha=0.7)
        ax.set_xlabel('Confidence Quartile')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Confidence Quartile')
        ax.set_xticks(x)
        ax.set_xticklabels(quartile_labels)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def batch_analyze_logs(self, log_directory: str, pattern: str = "*.json") -> Dict:
        """
        Analyze multiple log files and compare results across different models.
        
        Parameters:
        -----------
        log_directory : str
            Directory containing log files
        pattern : str
            File pattern to match
            
        Returns:
        --------
        Dict
            Aggregated results across all models
        """
        import glob
        
        log_files = glob.glob(os.path.join(log_directory, "**", pattern), recursive=True)
        
        all_results = {}
        
        print(f"Found {len(log_files)} log files to analyze...")
        
        for log_file in tqdm(log_files):
            try:
                # Extract model name from file path
                model_name = os.path.basename(os.path.dirname(log_file))
                
                # Run comparison
                results = self.compare_llm_to_optimal_bayesian_observer(
                    log_file, return_detailed_analysis=False
                )
                
                all_results[model_name] = results['summary_metrics']
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
                continue
        
        return all_results
    
    def plot_model_comparison(self, batch_results: Dict, save_path: str = None) -> None:
        """
        Create comparison plots across multiple models.
        
        Parameters:
        -----------
        batch_results : Dict
            Results from batch_analyze_logs
        save_path : str, optional
            Path to save the plot
        """
        if not batch_results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(batch_results.keys())
        
        # Extract metrics
        llm_accuracies = [batch_results[m]['llm_accuracy'] for m in models]
        optimal_accuracies = [batch_results[m]['optimal_accuracy'] for m in models]
        agreement_rates = [batch_results[m]['agreement_rate'] for m in models]
        performance_gaps = [batch_results[m]['performance_gap'] for m in models]
        
        # Plot 1: Accuracy comparison
        ax = axes[0, 0]
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, llm_accuracies, width, label='LLM', alpha=0.7)
        ax.bar(x + width/2, optimal_accuracies, width, label='Optimal', alpha=0.7)
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison Across Models')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        # Plot 2: Agreement rates
        ax = axes[0, 1]
        ax.bar(models, agreement_rates, alpha=0.7, color='lightgreen')
        ax.set_xlabel('Model')
        ax.set_ylabel('Agreement Rate')
        ax.set_title('LLM-Optimal Agreement Rates')
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Performance gaps
        ax = axes[1, 0]
        colors = ['red' if gap > 0 else 'green' for gap in performance_gaps]
        ax.bar(models, performance_gaps, alpha=0.7, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('Performance Gap')
        ax.set_title('Performance Gaps (Optimal - LLM)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: Scatter plot of accuracy vs agreement
        ax = axes[1, 1]
        ax.scatter(llm_accuracies, agreement_rates, alpha=0.7)
        for i, model in enumerate(models):
            ax.annotate(model, (llm_accuracies[i], agreement_rates[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.set_xlabel('LLM Accuracy')
        ax.set_ylabel('Agreement Rate')
        ax.set_title('LLM Accuracy vs Agreement Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
