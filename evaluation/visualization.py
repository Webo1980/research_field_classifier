"""
Research Field Classifier Visualization
=========================================
Generates comprehensive HTML reports with charts and per-paper analysis.

Features:
- Summary as TABLE comparing Old vs New approaches
- All visualization charts with proper data
- Per-paper details with FULL 5 predictions for BOTH classifiers
- ORKG scoring system with detailed explanations
- Combined report with all approaches comparison

Usage:
    python evaluation/visualization.py
"""

import json
import os
import sys
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import numpy as np
    HAS_MATPLOTLIB = True
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed")


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


class Visualizer:
    """Creates comprehensive visualizations and reports."""
    
    def __init__(self, results_dir: Path, reports_dir: Path):
        self.results_dir = results_dir
        self.reports_dir = reports_dir
        self.all_results: Dict[str, Dict] = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.colors = {
            'old': '#ff6b6b',
            'new': '#4ecdc4',
            'single_shot': '#2ecc71',
            'top_down': '#3498db',
            'embedding': '#9b59b6',
        }
        
        self._taxonomy_paths = {}
        self._load_taxonomy()
    
    def _load_taxonomy(self):
        """Load taxonomy for path display."""
        for path in [Path("./cache/taxonomy_tree.txt"), 
                     Path(settings.CACHE_DIR) / "taxonomy_tree.txt"]:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or not line.startswith('['):
                                continue
                            bracket_end = line.index(']')
                            field_id = line[1:bracket_end]
                            path_str = line[bracket_end + 1:].strip()
                            path_parts = [p.strip() for p in path_str.split('>')]
                            self._taxonomy_paths[field_id] = path_parts
                            if path_parts:
                                self._taxonomy_paths[path_parts[-1].lower()] = path_parts
                    logger.info(f"Loaded taxonomy paths")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load taxonomy: {e}")
    
    def _get_field_path(self, field_id: str = None, field_label: str = None) -> List[str]:
        """Get taxonomy path for a field."""
        if field_id and field_id in self._taxonomy_paths:
            return self._taxonomy_paths[field_id]
        if field_label:
            normalized = field_label.lower()
            if normalized in self._taxonomy_paths:
                return self._taxonomy_paths[normalized]
        return []
    
    def load_results(self):
        """Load evaluation results from all approaches."""
        approaches = ["single_shot", "top_down", "embedding"]
        
        for approach in approaches:
            approach_dir = self.results_dir / approach
            if not approach_dir.exists():
                logger.warning(f"No results directory for {approach}")
                continue
            
            json_files = list(approach_dir.glob("*_results_*.json"))
            if not json_files:
                logger.warning(f"No result files for {approach}")
                continue
            
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.all_results[approach] = data
                    logger.info(f"Loaded {approach}: {len(data.get('results', []))} papers")
                    
                    # Log metrics
                    m = data.get("metrics", {})
                    old_m = data.get("old_metrics", {})
                    logger.info(f"  NEW metrics: Top-1={m.get('top1_accuracy', 0)*100:.1f}%")
                    logger.info(f"  OLD metrics: Top-1={old_m.get('top1_accuracy', 0)*100:.1f}%")
            except Exception as e:
                logger.error(f"Failed to load {approach}: {e}")
        
        logger.info(f"Loaded {len(self.all_results)} approach results")
    
    # ========================================================================
    # CHART GENERATION
    # ========================================================================
    
    def create_accuracy_comparison_chart(self, approach: str) -> str:
        """Create accuracy comparison chart (OLD vs NEW)."""
        if not HAS_MATPLOTLIB or approach not in self.all_results:
            return ""
        
        data = self.all_results[approach]
        new_m = data.get("metrics", {})
        old_m = data.get("old_metrics", {})
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Position-based accuracy (ORKG)
        ax1 = axes[0]
        labels = ['Old Classifier\n(Flat)', f'New Classifier\n({approach.replace("_", " ").title()})']
        
        old_acc = old_m.get("accuracy_score", 0) * 100
        new_acc = new_m.get("accuracy_score", 0) * 100
        
        bars = ax1.bar(labels, [old_acc, new_acc], color=[self.colors['old'], self.colors['new']])
        ax1.set_ylabel('Accuracy Score (%)', fontsize=12)
        ax1.set_title('Position-Based Accuracy (ORKG System)\nRank-1=100%, Rank-2/3=80%, Rank-4/5=60%', 
                      fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, [old_acc, new_acc]):
            ax1.annotate(f'{val:.1f}%',
                        xy=(bar.get_x() + bar.get_width()/2, val + 2),
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Right: Binary match rates
        ax2 = axes[1]
        x = np.arange(2)
        width = 0.35
        
        old_vals = [old_m.get("top1_accuracy", 0) * 100, old_m.get("top5_accuracy", 0) * 100]
        new_vals = [new_m.get("top1_accuracy", 0) * 100, new_m.get("top5_accuracy", 0) * 100]
        
        bars1 = ax2.bar(x - width/2, old_vals, width, label='Old (Flat)', color=self.colors['old'])
        bars2 = ax2.bar(x + width/2, new_vals, width, 
                       label=f'New ({approach.replace("_", " ").title()})', color=self.colors['new'])
        
        ax2.set_ylabel('Match Rate (%)', fontsize=12)
        ax2.set_title('Binary Match Rates', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Top-1 Match', 'Top-5 Match'])
        ax2.legend(fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax2.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h + 2),
                                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_response_time_chart(self, approach: str) -> str:
        """Create response time comparison chart."""
        if not HAS_MATPLOTLIB or approach not in self.all_results:
            return ""
        
        results = self.all_results[approach].get("results", [])
        
        old_times = [r.get("old_response_time_ms", 0) for r in results if r.get("old_response_time_ms", 0) > 0]
        new_times = [r.get("response_time_ms", 0) for r in results if r.get("response_time_ms", 0) > 0]
        
        if not old_times and not new_times:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        plot_data = []
        labels = []
        colors_list = []
        
        if old_times:
            plot_data.append(old_times)
            labels.append("Old (Flat)")
            colors_list.append(self.colors['old'])
        
        if new_times:
            plot_data.append(new_times)
            labels.append(f"New ({approach.replace('_', ' ').title()})")
            colors_list.append(self.colors['new'])
        
        bp = ax.boxplot(plot_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for i, d in enumerate(plot_data):
            mean_val = np.mean(d)
            ax.scatter(i + 1, mean_val, marker='D', color='black', s=60, zorder=3)
            ax.annotate(f'Mean: {mean_val:.0f}ms',
                       xy=(i + 1, mean_val), xytext=(15, 0), textcoords="offset points",
                       fontsize=10, fontweight='bold')
        
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Response Time (ms)', fontsize=12)
        ax.set_title('Response Time Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_rank_distribution_chart(self, approach: str) -> str:
        """Create rank distribution comparison chart."""
        if not HAS_MATPLOTLIB or approach not in self.all_results:
            return ""
        
        results = self.all_results[approach].get("results", [])
        
        categories = ['Rank 1\n(100%)', 'Rank 2\n(80%)', 'Rank 3\n(80%)', 
                     'Rank 4\n(60%)', 'Rank 5\n(60%)', 'Not Found\n(0%)']
        
        old_ranks = [0] * 6
        new_ranks = [0] * 6
        
        for r in results:
            old_pos = r.get("old_match_position", 0)
            new_pos = r.get("match_position", 0)
            
            if 1 <= old_pos <= 5:
                old_ranks[old_pos - 1] += 1
            else:
                old_ranks[5] += 1
            
            if 1 <= new_pos <= 5:
                new_ranks[new_pos - 1] += 1
            else:
                new_ranks[5] += 1
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, old_ranks, width, label='Old (Flat)', color=self.colors['old'])
        bars2 = ax.bar(x + width/2, new_ranks, width, 
                      label=f'New ({approach.replace("_", " ").title()})', color=self.colors['new'])
        
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title('Rank Distribution Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(str(int(h)), xy=(bar.get_x() + bar.get_width()/2, h + 0.2),
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_match_heatmap_chart(self, approach: str) -> str:
        """Create classification match heatmap (OLD vs NEW)."""
        if not HAS_MATPLOTLIB or approach not in self.all_results:
            return ""
        
        data = self.all_results[approach]
        results = data.get("results", [])
        old_m = data.get("old_metrics", {})
        new_m = data.get("metrics", {})
        
        if not results:
            return ""
        
        n = len(results)
        paper_ids = [r.get("paper_id", f"P{i}")[:10] for i, r in enumerate(results)]
        
        # Create match matrices
        old_matrix = np.zeros((5, n))
        new_matrix = np.zeros((5, n))
        
        for i, r in enumerate(results):
            gt_label = r.get("ground_truth_label", "").lower()
            
            # OLD
            old_preds = r.get("old_predictions", [])
            for rank, pred in enumerate(old_preds[:5]):
                pred_label = pred.get("field", "").lower()
                h_dist = pred.get("h_distance", -1)
                if pred_label and (pred_label == gt_label or h_dist == 0):
                    old_matrix[rank, i] = 1
                    break
            
            # NEW
            new_preds = r.get("predictions", [])
            for rank, pred in enumerate(new_preds[:5]):
                pred_label = pred.get("field", "").lower()
                h_dist = pred.get("h_distance", -1)
                if pred_label and (pred_label == gt_label or h_dist == 0):
                    new_matrix[rank, i] = 1
                    break
        
        fig_width = max(16, n * 0.5)
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, 7))
        
        cmap = ListedColormap(['#ffcccc', '#28a745'])
        
        # OLD heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(old_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax1.set_yticks(range(5))
        ax1.set_yticklabels([f'Rank {i+1}' for i in range(5)])
        ax1.set_xticks(range(n))
        ax1.set_xticklabels(paper_ids, rotation=90, fontsize=7)
        ax1.set_xlabel('Paper ID')
        
        for i in range(5):
            for j in range(n):
                val = int(old_matrix[i, j])
                ax1.text(j, i, str(val), ha='center', va='center',
                        color='white' if val == 1 else 'gray', fontsize=7, fontweight='bold')
        
        ax1.set_title(f'OLD Classifier (Flat)\nTop-1: {old_m.get("top1_accuracy", 0)*100:.1f}% | Top-5: {old_m.get("top5_accuracy", 0)*100:.1f}%',
                     fontsize=11, fontweight='bold', color=self.colors['old'])
        
        # NEW heatmap
        ax2 = axes[1]
        im2 = ax2.imshow(new_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels([f'Rank {i+1}' for i in range(5)])
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(paper_ids, rotation=90, fontsize=7)
        ax2.set_xlabel('Paper ID')
        
        for i in range(5):
            for j in range(n):
                val = int(new_matrix[i, j])
                ax2.text(j, i, str(val), ha='center', va='center',
                        color='white' if val == 1 else 'gray', fontsize=7, fontweight='bold')
        
        ax2.set_title(f'NEW Classifier ({approach.replace("_", " ").title()})\nTop-1: {new_m.get("top1_accuracy", 0)*100:.1f}% | Top-5: {new_m.get("top5_accuracy", 0)*100:.1f}%',
                     fontsize=11, fontweight='bold', color=self.colors['new'])
        
        # Colorbar
        cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.015, pad=0.02)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['No Match', 'Match'])
        
        plt.suptitle('Classification Match Heatmap: OLD vs NEW', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_combined_chart(self) -> str:
        """Create combined comparison chart for all approaches."""
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect data
        names = []
        top1_vals = []
        top5_vals = []
        acc_score_vals = []
        time_vals = []
        colors_list = []
        
        # Add OLD first (from first available approach)
        first_approach = list(self.all_results.keys())[0]
        old_m = self.all_results[first_approach].get("old_metrics", {})
        
        names.append("Old\n(Flat)")
        top1_vals.append(old_m.get("top1_accuracy", 0) * 100)
        top5_vals.append(old_m.get("top5_accuracy", 0) * 100)
        acc_score_vals.append(old_m.get("accuracy_score", 0) * 100)
        time_vals.append(old_m.get("avg_response_time_ms", 0))
        colors_list.append(self.colors['old'])
        
        # Add NEW approaches
        for approach in self.all_results:
            m = self.all_results[approach].get("metrics", {})
            names.append(approach.replace("_", "\n").title())
            top1_vals.append(m.get("top1_accuracy", 0) * 100)
            top5_vals.append(m.get("top5_accuracy", 0) * 100)
            acc_score_vals.append(m.get("accuracy_score", 0) * 100)
            time_vals.append(m.get("avg_response_time_ms", 0))
            colors_list.append(self.colors.get(approach, '#4ecdc4'))
        
        x = np.arange(len(names))
        
        # Top-1 Accuracy
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x, top1_vals, color=colors_list)
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('Top-1 Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylim(0, max(100, max(top1_vals) * 1.2) if max(top1_vals) > 0 else 100)
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars1, top1_vals):
            if val > 0:
                ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val + 1),
                            ha='center', fontweight='bold')
        
        # Top-5 Accuracy
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x, top5_vals, color=colors_list)
        ax2.set_ylabel('Top-5 Accuracy (%)')
        ax2.set_title('Top-5 Accuracy Comparison', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.set_ylim(0, max(100, max(top5_vals) * 1.2) if max(top5_vals) > 0 else 100)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars2, top5_vals):
            if val > 0:
                ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val + 1),
                            ha='center', fontweight='bold')
        
        # Accuracy Score (ORKG weighted)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(x, acc_score_vals, color=colors_list)
        ax3.set_ylabel('Accuracy Score (%)')
        ax3.set_title('ORKG Position-Based Accuracy Score', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names)
        ax3.set_ylim(0, max(100, max(acc_score_vals) * 1.2) if max(acc_score_vals) > 0 else 100)
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars3, acc_score_vals):
            if val > 0:
                ax3.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val + 1),
                            ha='center', fontweight='bold')
        
        # Response Time
        ax4 = axes[1, 1]
        bars4 = ax4.bar(x, time_vals, color=colors_list)
        ax4.set_ylabel('Avg Response Time (ms)')
        ax4.set_title('Response Time Comparison', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(names)
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars4, time_vals):
            if val > 0:
                ax4.annotate(f'{val:.0f}', xy=(bar.get_x() + bar.get_width()/2, val + 50),
                            ha='center', fontweight='bold')
        
        plt.suptitle('Research Field Classifier - All Approaches Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_radar_chart(self) -> str:
        """
        Create radar chart comparing all approaches across multiple dimensions.
        
        Dimensions:
        - Top-1 Accuracy
        - Top-5 Accuracy
        - ORKG Score
        - Speed (inverted - faster is better)
        - Token Efficiency (inverted - fewer is better)
        - H-Distance (inverted - lower is better)
        """
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        # Collect metrics for all approaches
        approaches = []
        metrics_data = []
        
        # Add OLD first
        first_approach = list(self.all_results.keys())[0]
        old_m = self.all_results[first_approach].get("old_metrics", {})
        
        approaches.append("Old (Flat)")
        metrics_data.append({
            'top1': old_m.get("top1_accuracy", 0) * 100,
            'top5': old_m.get("top5_accuracy", 0) * 100,
            'orkg_score': old_m.get("accuracy_score", 0) * 100,
            'response_time': old_m.get("avg_response_time_ms", 0),
            'tokens': 0,  # Old classifier doesn't use tokens
            'h_distance': old_m.get("avg_h_distance", 10)
        })
        
        # Add new approaches
        for approach in self.all_results:
            m = self.all_results[approach].get("metrics", {})
            approaches.append(approach.replace("_", " ").title())
            metrics_data.append({
                'top1': m.get("top1_accuracy", 0) * 100,
                'top5': m.get("top5_accuracy", 0) * 100,
                'orkg_score': m.get("accuracy_score", 0) * 100,
                'response_time': m.get("avg_response_time_ms", 0),
                'tokens': m.get("total_tokens", 0),
                'h_distance': m.get("avg_h_distance", 10)
            })
        
        # Normalize metrics to 0-100 scale for radar chart
        # For "lower is better" metrics, invert the scale
        max_time = max(d['response_time'] for d in metrics_data) or 1
        max_tokens = max(d['tokens'] for d in metrics_data) or 1
        max_h_dist = max(d['h_distance'] for d in metrics_data) or 1
        
        categories = ['Top-1\nAccuracy', 'Top-5\nAccuracy', 'ORKG\nScore', 
                      'Speed', 'Token\nEfficiency', 'Proximity\n(H-Dist)']
        N = len(categories)
        
        # Compute normalized values for each approach
        normalized_data = []
        for d in metrics_data:
            normalized = [
                d['top1'],
                d['top5'],
                d['orkg_score'],
                100 - (d['response_time'] / max_time * 100) if max_time > 0 else 100,  # Inverted
                100 - (d['tokens'] / max_tokens * 100) if max_tokens > 0 and d['tokens'] > 0 else 100,  # Inverted
                100 - (d['h_distance'] / max_h_dist * 100) if max_h_dist > 0 else 100  # Inverted
            ]
            normalized_data.append(normalized)
        
        # Create radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = [self.colors['old']] + [self.colors.get(a, '#4ecdc4') for a in self.all_results.keys()]
        
        for i, (approach, data, color) in enumerate(zip(approaches, normalized_data, colors)):
            values = data + data[:1]  # Complete the loop
            ax.plot(angles, values, 'o-', linewidth=2, label=approach, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.title('Multi-Dimensional Approach Comparison\n(Higher is Better for All Axes)', 
                  fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_combined_rank_distribution(self) -> str:
        """Create combined rank distribution chart for all approaches."""
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        categories = ['Rank 1\n(100%)', 'Rank 2\n(80%)', 'Rank 3\n(80%)', 
                     'Rank 4\n(60%)', 'Rank 5\n(60%)', 'Not Found\n(0%)']
        
        # Collect rank distributions
        all_distributions = {}
        
        # Get OLD distribution from first approach
        first_approach = list(self.all_results.keys())[0]
        results = self.all_results[first_approach].get("results", [])
        
        old_ranks = [0] * 6
        for r in results:
            old_pos = r.get("old_match_position", 0)
            if 1 <= old_pos <= 5:
                old_ranks[old_pos - 1] += 1
            else:
                old_ranks[5] += 1
        all_distributions["Old (Flat)"] = old_ranks
        
        # Get NEW distributions for each approach
        for approach in self.all_results:
            results = self.all_results[approach].get("results", [])
            new_ranks = [0] * 6
            for r in results:
                new_pos = r.get("match_position", 0)
                if 1 <= new_pos <= 5:
                    new_ranks[new_pos - 1] += 1
                else:
                    new_ranks[5] += 1
            all_distributions[approach.replace("_", " ").title()] = new_ranks
        
        # Plot grouped bars
        x = np.arange(len(categories))
        n_approaches = len(all_distributions)
        width = 0.8 / n_approaches
        
        colors = [self.colors['old']] + [self.colors.get(a, '#4ecdc4') for a in self.all_results.keys()]
        
        for i, (name, ranks) in enumerate(all_distributions.items()):
            offset = (i - n_approaches / 2 + 0.5) * width
            bars = ax.bar(x + offset, ranks, width, label=name, color=colors[i], alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, ranks):
                if val > 0:
                    ax.annotate(f'{val}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1),
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title('Rank Distribution Comparison - All Approaches', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_response_time_per_paper(self) -> str:
        """Create response time comparison per paper for all approaches."""
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        # Get paper IDs from first approach
        first_approach = list(self.all_results.keys())[0]
        results = self.all_results[first_approach].get("results", [])
        paper_ids = [r.get("paper_id", f"P{i}")[:8] for i, r in enumerate(results)]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(paper_ids))
        n_approaches = len(self.all_results) + 1  # +1 for old classifier
        width = 0.8 / n_approaches
        
        # Plot OLD classifier times
        old_times = [r.get("old_response_time_ms", 0) for r in results]
        offset = (0 - n_approaches / 2 + 0.5) * width
        ax.bar(x + offset, old_times, width, label='Old (Flat)', color=self.colors['old'], alpha=0.8)
        
        # Plot NEW classifier times for each approach
        for i, approach in enumerate(self.all_results.keys(), 1):
            results = self.all_results[approach].get("results", [])
            new_times = [r.get("response_time_ms", 0) for r in results]
            offset = (i - n_approaches / 2 + 0.5) * width
            color = self.colors.get(approach, '#4ecdc4')
            ax.bar(x + offset, new_times, width, label=approach.replace("_", " ").title(), 
                   color=color, alpha=0.8)
        
        ax.set_ylabel('Response Time (ms)', fontsize=12)
        ax.set_xlabel('Paper ID', fontsize=12)
        ax.set_title('Response Time per Paper - All Approaches', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(paper_ids, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_timing_breakdown_chart(self) -> str:
        """Create timing breakdown chart showing LLM time, embedding time, etc."""
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Stacked bar chart of timing components
        ax1 = axes[0]
        
        approaches = list(self.all_results.keys())
        llm_times = []
        other_times = []
        
        for approach in approaches:
            results = self.all_results[approach].get("results", [])
            
            # Calculate average timing breakdown
            avg_llm = np.mean([r.get("llm_time_ms", r.get("response_time_ms", 0) * 0.9) for r in results])
            avg_total = np.mean([r.get("response_time_ms", 0) for r in results])
            avg_other = avg_total - avg_llm
            
            llm_times.append(avg_llm)
            other_times.append(max(0, avg_other))
        
        x = np.arange(len(approaches))
        width = 0.6
        
        colors_approaches = [self.colors.get(a, '#4ecdc4') for a in approaches]
        
        bars1 = ax1.bar(x, llm_times, width, label='LLM Processing', color=colors_approaches, alpha=0.9)
        bars2 = ax1.bar(x, other_times, width, bottom=llm_times, label='Other (Network, Parsing)', 
                       color='#95a5a6', alpha=0.5, hatch='//')
        
        ax1.set_ylabel('Time (ms)', fontsize=12)
        ax1.set_title('Timing Breakdown by Approach', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a.replace("_", " ").title() for a in approaches], fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add total time labels
        for i, (llm, other) in enumerate(zip(llm_times, other_times)):
            total = llm + other
            ax1.annotate(f'{total:.0f}ms', xy=(i, total + 100), ha='center', 
                        fontsize=10, fontweight='bold')
        
        # Right: Pie chart of average timing distribution
        ax2 = axes[1]
        
        # Combine all approaches for overall timing
        all_llm = np.mean(llm_times)
        all_other = np.mean(other_times)
        
        # Get old classifier time for comparison
        first_approach = list(self.all_results.keys())[0]
        old_m = self.all_results[first_approach].get("old_metrics", {})
        old_time = old_m.get("avg_response_time_ms", 0)
        
        if old_time > 0:
            sizes = [old_time, all_llm, all_other]
            labels = [f'Old Classifier\n({old_time:.0f}ms)', 
                     f'New LLM Processing\n({all_llm:.0f}ms)', 
                     f'New Other\n({all_other:.0f}ms)']
            colors_pie = [self.colors['old'], self.colors['new'], '#95a5a6']
        else:
            sizes = [all_llm, all_other]
            labels = [f'LLM Processing\n({all_llm:.0f}ms)', 
                     f'Other\n({all_other:.0f}ms)']
            colors_pie = [self.colors['new'], '#95a5a6']
        
        wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie,
                                            autopct='%1.1f%%', startangle=90,
                                            explode=[0.02] * len(sizes))
        
        ax2.set_title('Average Time Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_token_usage_chart(self) -> str:
        """Create token usage comparison chart for all approaches."""
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Total tokens per approach
        ax1 = axes[0]
        
        approaches = list(self.all_results.keys())
        total_tokens = []
        avg_input = []
        avg_output = []
        
        for approach in approaches:
            m = self.all_results[approach].get("metrics", {})
            results = self.all_results[approach].get("results", [])
            
            total = m.get("total_tokens", 0)
            if total == 0:
                # Calculate from results
                total = sum(r.get("total_tokens", 0) for r in results)
            total_tokens.append(total)
            
            # Calculate averages
            inputs = [r.get("input_tokens", 0) for r in results]
            outputs = [r.get("output_tokens", 0) for r in results]
            avg_input.append(np.mean(inputs) if inputs else 0)
            avg_output.append(np.mean(outputs) if outputs else 0)
        
        x = np.arange(len(approaches))
        colors_approaches = [self.colors.get(a, '#4ecdc4') for a in approaches]
        
        bars = ax1.bar(x, total_tokens, color=colors_approaches, alpha=0.8)
        ax1.set_ylabel('Total Tokens', fontsize=12)
        ax1.set_title('Total Token Usage by Approach', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a.replace("_", " ").title() for a in approaches], fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, total_tokens):
            if val > 0:
                ax1.annotate(f'{val:,}', xy=(bar.get_x() + bar.get_width()/2, val + 500),
                            ha='center', fontsize=10, fontweight='bold')
        
        # Right: Input vs Output tokens (stacked)
        ax2 = axes[1]
        
        width = 0.6
        bars1 = ax2.bar(x, avg_input, width, label='Input Tokens (avg)', 
                       color=colors_approaches, alpha=0.9)
        bars2 = ax2.bar(x, avg_output, width, bottom=avg_input, label='Output Tokens (avg)',
                       color=colors_approaches, alpha=0.5, hatch='//')
        
        ax2.set_ylabel('Tokens per Paper (avg)', fontsize=12)
        ax2.set_title('Token Distribution: Input vs Output', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([a.replace("_", " ").title() for a in approaches], fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    def create_hdistance_comparison_chart(self) -> str:
        """Create H-Distance comparison chart for all approaches."""
        if not HAS_MATPLOTLIB or not self.all_results:
            return ""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Average H-Distance
        ax1 = axes[0]
        
        approaches = ["Old (Flat)"] + list(self.all_results.keys())
        avg_distances = []
        
        # Old classifier
        first_approach = list(self.all_results.keys())[0]
        old_m = self.all_results[first_approach].get("old_metrics", {})
        avg_distances.append(old_m.get("avg_h_distance", 0))
        
        # New approaches
        for approach in self.all_results:
            m = self.all_results[approach].get("metrics", {})
            avg_distances.append(m.get("avg_h_distance", 0))
        
        x = np.arange(len(approaches))
        colors_list = [self.colors['old']] + [self.colors.get(a, '#4ecdc4') for a in self.all_results.keys()]
        
        bars = ax1.bar(x, avg_distances, color=colors_list, alpha=0.8)
        ax1.set_ylabel('Average H-Distance (lower is better)', fontsize=12)
        ax1.set_title('Hierarchical Distance Comparison', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([a.replace("_", " ").title() if "_" in a else a for a in approaches], 
                           fontsize=10, rotation=15)
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, avg_distances):
            if val > 0:
                ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val + 0.1),
                            ha='center', fontsize=10, fontweight='bold')
        
        # Right: H-Distance distribution (box plot)
        ax2 = axes[1]
        
        all_h_dists = []
        labels = []
        colors_box = []
        
        # Get H-distances from results
        first_results = self.all_results[first_approach].get("results", [])
        old_dists = [r.get("old_predictions", [{}])[0].get("h_distance", -1) 
                    for r in first_results if r.get("old_predictions")]
        old_dists = [d for d in old_dists if d >= 0]
        if old_dists:
            all_h_dists.append(old_dists)
            labels.append("Old (Flat)")
            colors_box.append(self.colors['old'])
        
        for approach in self.all_results:
            results = self.all_results[approach].get("results", [])
            dists = [r.get("h_distance", -1) for r in results]
            dists = [d for d in dists if d >= 0]
            if dists:
                all_h_dists.append(dists)
                labels.append(approach.replace("_", " ").title())
                colors_box.append(self.colors.get(approach, '#4ecdc4'))
        
        if all_h_dists:
            bp = ax2.boxplot(all_h_dists, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors_box):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_xticklabels(labels, fontsize=10, rotation=15)
            ax2.set_ylabel('H-Distance', fontsize=12)
            ax2.set_title('H-Distance Distribution', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        img = fig_to_base64(fig)
        plt.close()
        return img
    
    # ========================================================================
    # HTML GENERATION
    # ========================================================================
    
    def get_css(self) -> str:
        """Return CSS styles."""
        return """
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
.container { max-width: 1400px; margin: 0 auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
h1 { color: #333; border-bottom: 3px solid #4ecdc4; padding-bottom: 10px; }
h2 { color: #4ecdc4; margin-top: 30px; }
h3, h4 { color: #666; margin: 15px 0 10px 0; }

/* Summary Table */
.summary-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; }
.summary-table th { background: #4ecdc4; color: white; padding: 12px 15px; text-align: left; font-weight: 600; }
.summary-table th.old { background: #ff6b6b; }
.summary-table td { padding: 12px 15px; border-bottom: 1px solid #eee; }
.summary-table tr:hover { background: #f8f9fa; }
.summary-table .metric-name { font-weight: 600; color: #555; }
.summary-table .value-old { color: #ff6b6b; font-weight: bold; }
.summary-table .value-new { color: #4ecdc4; font-weight: bold; }
.summary-table .best { background: #d4edda; }

/* Charts */
.chart-section { margin: 20px 0; }
.charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
@media (max-width: 1000px) { .charts-grid { grid-template-columns: 1fr; } }
.chart-box { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
.chart-box img { max-width: 100%; display: block; margin: 0 auto; }
.chart-box.full { grid-column: span 2; }
@media (max-width: 1000px) { .chart-box.full { grid-column: span 1; } }

/* Paper cards */
.paper-card { border: 1px solid #e0e0e0; border-radius: 8px; margin: 12px 0; overflow: hidden; }
.paper-header { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: #f8f9fa; cursor: pointer; }
.paper-header:hover { background: #e9ecef; }
.paper-header.matched { background: #d4edda; border-left: 4px solid #28a745; }
.paper-header.matched-partial { background: #cce5ff; border-left: 4px solid #007bff; }
.paper-header.not-matched { background: #fff3cd; border-left: 4px solid #ffc107; }
.paper-title { font-weight: 600; color: #333; flex: 1; }
.status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; font-weight: 500; margin-left: 10px; }
.status-matched { background: #28a745; color: white; }
.status-matched-partial { background: #007bff; color: white; }
.status-not-matched { background: #ffc107; color: #333; }

.paper-details { padding: 20px; background: #fafafa; display: none; }
.paper-details.show { display: block; }

.detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 15px 0; }
.detail-item { background: white; padding: 12px; border-radius: 6px; border: 1px solid #eee; }
.detail-label { font-size: 0.8em; color: #888; text-transform: uppercase; margin-bottom: 4px; }
.detail-value { font-weight: 600; color: #333; word-break: break-word; }

.path-section { margin: 20px 0; }
.path-box { background: #f8f9fa; border-radius: 6px; padding: 12px; margin: 8px 0; border-left: 3px solid #6c757d; }
.path-box.gt { border-left-color: #dc3545; background: #fff5f5; }
.path-box.pred { border-left-color: #28a745; background: #f0fff4; }
.path-label { font-weight: 600; font-size: 0.9em; color: #666; margin-bottom: 4px; }
.path-chain { font-family: 'Courier New', monospace; font-size: 0.9em; color: #333; }

.abstract-box { background: #f0f4f8; border-radius: 6px; padding: 12px; margin: 15px 0; max-height: 120px; overflow-y: auto; font-size: 0.9em; line-height: 1.6; color: #555; }

.predictions-section { margin: 20px 0; }
.pred-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
@media (max-width: 900px) { .pred-grid { grid-template-columns: 1fr; } }
.pred-box { border-radius: 8px; padding: 15px; }
.pred-box.old { border: 2px solid #ff6b6b; background: #fff8f8; }
.pred-box.new { border: 2px solid #4ecdc4; background: #f0fffd; }
.pred-box h5 { margin: 0 0 10px 0; }
.pred-box.old h5 { color: #ff6b6b; }
.pred-box.new h5 { color: #4ecdc4; }

.pred-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
.pred-table th { background: #6c757d; color: white; padding: 10px 8px; text-align: left; font-size: 0.85em; }
.pred-table td { padding: 10px 8px; border-bottom: 1px solid #eee; }
.pred-table tr:hover { background: rgba(0,0,0,0.02); }
.pred-table tr.match { background: #d4edda !important; }
.pred-table .field-id { font-family: monospace; font-size: 0.8em; color: #666; }
.match-icon { background: #28a745; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-left: 5px; }

.dist-0 { color: #28a745; font-weight: bold; }
.dist-1, .dist-2 { color: #ffc107; font-weight: 600; }
.dist-3, .dist-4, .dist-5, .dist-6, .dist-7 { color: #dc3545; }
.dist-na { color: #6c757d; font-style: italic; }

.legend-box { background: #f9f9f9; padding: 20px; border-radius: 8px; margin: 20px 0; border: 1px solid #e0e0e0; }
.legend-box h4 { margin: 0 0 15px 0; color: #333; }
.legend-box ul { margin: 10px 0; padding-left: 25px; }
.legend-box li { margin: 8px 0; }
.legend-box p { margin: 10px 0; color: #666; }

.btn { background: #4ecdc4; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin: 5px 5px 15px 0; font-weight: 500; }
.btn:hover { background: #3dbdb5; }

.comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
.comparison-table th, .comparison-table td { padding: 12px 15px; border: 1px solid #ddd; text-align: left; }
.comparison-table th { background: #4ecdc4; color: white; }
.comparison-table tr:nth-child(even) { background: #f8f9fa; }
.comparison-table tr.best { background: #d4edda; }
.comparison-table .old-row { background: #fff0f0; }

.toggle-icon { display: inline-block; width: 20px; font-weight: bold; }

.no-data { color: #999; font-style: italic; }
</style>
<script>
function toggleDetails(id) {
    var d = document.getElementById('details-' + id);
    var i = document.getElementById('icon-' + id);
    if (d.classList.contains('show')) {
        d.classList.remove('show');
        i.textContent = '+';
    } else {
        d.classList.add('show');
        i.textContent = '−';
    }
}
function toggleAll(show) {
    document.querySelectorAll('.paper-details').forEach(function(d) { d.classList.toggle('show', show); });
    document.querySelectorAll('.toggle-icon').forEach(function(i) { i.textContent = show ? '−' : '+'; });
}
</script>
"""
    
    def generate_approach_report(self, approach: str) -> str:
        """Generate HTML report for one approach."""
        if approach not in self.all_results:
            return ""
        
        data = self.all_results[approach]
        new_m = data.get("metrics", {})
        old_m = data.get("old_metrics", {})
        results = data.get("results", [])
        
        charts = {
            'accuracy': self.create_accuracy_comparison_chart(approach),
            'time': self.create_response_time_chart(approach),
            'rank': self.create_rank_distribution_chart(approach),
            'heatmap': self.create_match_heatmap_chart(approach),
        }
        
        approach_title = approach.replace("_", " ").title()
        
        # Determine best values for highlighting
        best_top1 = max(old_m.get("top1_accuracy", 0), new_m.get("top1_accuracy", 0))
        best_top5 = max(old_m.get("top5_accuracy", 0), new_m.get("top5_accuracy", 0))
        best_acc = max(old_m.get("accuracy_score", 0), new_m.get("accuracy_score", 0))
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{approach_title} Classifier - Evaluation Report</title>
    <meta charset="UTF-8">
    {self.get_css()}
</head>
<body>
<div class="container">
    <h1>📊 {approach_title} Classifier - Evaluation Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Papers Evaluated:</strong> {len(results)}</p>
    
    <h2>Summary Comparison</h2>
    <table class="summary-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th class="old">Old Classifier (Flat)</th>
                <th>New Classifier ({approach_title})</th>
            </tr>
        </thead>
        <tbody>
            <tr class="{'best' if old_m.get('accuracy_score', 0) >= new_m.get('accuracy_score', 0) and old_m.get('accuracy_score', 0) > 0 else ''}">
                <td class="metric-name">Accuracy Score (ORKG Weighted)</td>
                <td class="value-old">{old_m.get('accuracy_score', 0)*100:.1f}%</td>
                <td class="value-new {'best' if new_m.get('accuracy_score', 0) > old_m.get('accuracy_score', 0) else ''}">{new_m.get('accuracy_score', 0)*100:.1f}%</td>
            </tr>
            <tr>
                <td class="metric-name">Top-1 Accuracy (Exact Match)</td>
                <td class="value-old">{old_m.get('top1_accuracy', 0)*100:.1f}%</td>
                <td class="value-new">{new_m.get('top1_accuracy', 0)*100:.1f}%</td>
            </tr>
            <tr>
                <td class="metric-name">Top-5 Accuracy (Match in Top 5)</td>
                <td class="value-old">{old_m.get('top5_accuracy', 0)*100:.1f}%</td>
                <td class="value-new">{new_m.get('top5_accuracy', 0)*100:.1f}%</td>
            </tr>
            <tr>
                <td class="metric-name">Average Response Time</td>
                <td class="value-old">{old_m.get('avg_response_time_ms', 0):.0f} ms</td>
                <td class="value-new">{new_m.get('avg_response_time_ms', 0):.0f} ms</td>
            </tr>
            <tr>
                <td class="metric-name">Average H-Distance (Top-1)</td>
                <td class="value-old">{old_m.get('avg_h_distance', 0):.2f}</td>
                <td class="value-new">{new_m.get('avg_h_distance', 0):.2f}</td>
            </tr>
            <tr>
                <td class="metric-name">Total Tokens Used</td>
                <td class="value-old">N/A</td>
                <td class="value-new">{new_m.get('total_tokens', 0):,}</td>
            </tr>
        </tbody>
    </table>
    
    <h2>📈 Visualizations</h2>
    <div class="chart-section">
"""
        
        if charts['accuracy']:
            html += f'<div class="chart-box full"><h4>Accuracy Comparison</h4><img src="data:image/png;base64,{charts["accuracy"]}"></div>'
        
        html += '<div class="charts-grid">'
        if charts['time']:
            html += f'<div class="chart-box"><h4>Response Time Distribution</h4><img src="data:image/png;base64,{charts["time"]}"></div>'
        if charts['rank']:
            html += f'<div class="chart-box"><h4>Rank Distribution</h4><img src="data:image/png;base64,{charts["rank"]}"></div>'
        html += '</div>'
        
        if charts['heatmap']:
            html += f'<div class="chart-box full"><h4>Classification Match Heatmap (OLD vs NEW)</h4><img src="data:image/png;base64,{charts["heatmap"]}"></div>'
        
        html += """
    </div>
    
    <h2>📋 Per-Paper Detailed Results</h2>
    <div>
        <button class="btn" onclick="toggleAll(true)">Expand All</button>
        <button class="btn" onclick="toggleAll(false)">Collapse All</button>
    </div>
"""
        
        for i, r in enumerate(results):
            paper_id = r.get("paper_id", f"P{i}")
            new_matched_top1 = r.get("correct_top1", False)
            new_matched_top5 = r.get("correct_top5", False)
            old_matched_top1 = r.get("old_correct_top1", False)
            old_matched_top5 = r.get("old_correct_top5", False)
            new_match_pos = r.get("match_position", 0)
            old_match_pos = r.get("old_match_position", 0)
            
            # Determine status text and styling
            if new_matched_top1:
                header_class = "matched"
                status_class = "status-matched"
                status_text = "NEW Top-1 ✓"
            elif new_matched_top5:
                header_class = "matched-partial"  # Partial match styling
                status_class = "status-matched-partial"
                status_text = f"Matched at Rank {new_match_pos}"
            elif old_matched_top1:
                header_class = "not-matched"
                status_class = "status-not-matched"
                status_text = "OLD Top-1 ✓"
            elif old_matched_top5:
                header_class = "not-matched"
                status_class = "status-not-matched"
                status_text = f"OLD Rank {old_match_pos}"
            else:
                header_class = "not-matched"
                status_class = "status-not-matched"
                status_text = "Not in Top-5"
            
            gt_label = r.get("ground_truth_label", "N/A")
            gt_id = r.get("ground_truth_id", "")
            gt_path = r.get("ground_truth_path", [])
            if not gt_path:
                gt_path = self._get_field_path(gt_id, gt_label)
            
            new_preds = r.get("predictions", [])
            old_preds = r.get("old_predictions", [])
            
            # Get top-1 for display
            new_top1 = new_preds[0].get("field", "") if new_preds and new_preds[0].get("field") else "N/A"
            old_top1 = old_preds[0].get("field", "") if old_preds and old_preds[0].get("field") else "N/A"
            
            new_top1_path = new_preds[0].get("path", []) if new_preds else []
            
            h_distance = r.get("h_distance", -1)
            if h_distance < 0 and new_preds and new_preds[0].get("h_distance", -1) >= 0:
                h_distance = new_preds[0].get("h_distance")
            
            title_short = r.get('title', 'No Title')[:80]
            if len(r.get('title', '')) > 80:
                title_short += '...'
            
            html += f"""
    <div class="paper-card">
        <div class="paper-header {header_class}" onclick="toggleDetails('{paper_id}')">
            <span class="paper-title">
                <span id="icon-{paper_id}" class="toggle-icon">+</span>
                {paper_id} - {title_short}
            </span>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
        <div id="details-{paper_id}" class="paper-details">
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Ground Truth</div>
                    <div class="detail-value">{gt_label}<br><small>({gt_id})</small></div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">New Top-1</div>
                    <div class="detail-value">{new_top1 if new_top1 != "N/A" else '<span class="no-data">No prediction</span>'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Old Top-1</div>
                    <div class="detail-value">{old_top1}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">New H-Distance</div>
                    <div class="detail-value">{h_distance if h_distance >= 0 else 'N/A'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">New Response Time</div>
                    <div class="detail-value">{r.get('response_time_ms', 0):.0f} ms</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Old Response Time</div>
                    <div class="detail-value">{r.get('old_response_time_ms', 0):.0f} ms</div>
                </div>
            </div>
            
            <div class="abstract-box">
                <strong>Abstract:</strong> {r.get('abstract', 'No abstract available')}
            </div>
            
            <div class="path-section">
                <h4>📍 Taxonomy Paths</h4>
                <div class="path-box gt">
                    <div class="path-label">🎯 Ground Truth Path:</div>
                    <div class="path-chain">{' › '.join(gt_path) if gt_path else 'Path not available'}</div>
                </div>
                <div class="path-box pred">
                    <div class="path-label">🔮 New Top-1 Prediction Path:</div>
                    <div class="path-chain">{' › '.join(new_top1_path) if new_top1_path else '<span class="no-data">No prediction path</span>'}</div>
                </div>
            </div>
            
            <div class="predictions-section">
                <h4>📊 Full Predictions Comparison (All 5 Ranks)</h4>
                <div class="pred-grid">
                    <div class="pred-box old">
                        <h5>🔴 OLD Classifier (Flat)</h5>
                        <table class="pred-table">
                            <thead><tr><th>Rank</th><th>Predicted Field</th><th>Field ID</th><th>Score</th><th>H-Dist</th></tr></thead>
                            <tbody>
"""
            
            # OLD predictions (all 5)
            for rank in range(5):
                if rank < len(old_preds) and old_preds[rank].get("field"):
                    p = old_preds[rank]
                    field = p.get("field", "")
                    fid = p.get("field_id", "") or "N/A"
                    score = p.get("score", 0)
                    hdist = p.get("h_distance", -1)
                    
                    is_match = field.lower() == gt_label.lower() or hdist == 0
                    row_class = "match" if is_match else ""
                    dist_class = f"dist-{hdist}" if hdist >= 0 else "dist-na"
                    match_icon = '<span class="match-icon">✓</span>' if is_match else ""
                    hdist_display = str(hdist) if hdist >= 0 else "N/A"
                    
                    html += f'<tr class="{row_class}"><td><strong>{rank+1}</strong></td><td>{field}{match_icon}</td><td class="field-id">{fid}</td><td>{score:.2f}</td><td class="{dist_class}">{hdist_display}</td></tr>'
                else:
                    html += f'<tr><td>{rank+1}</td><td colspan="4" class="no-data">No prediction</td></tr>'
            
            html += f"""
                            </tbody>
                        </table>
                    </div>
                    <div class="pred-box new">
                        <h5>🟢 NEW Classifier ({approach_title})</h5>
                        <table class="pred-table">
                            <thead><tr><th>Rank</th><th>Predicted Field</th><th>Field ID</th><th>Score</th><th>H-Dist</th></tr></thead>
                            <tbody>
"""
            
            # NEW predictions (all 5)
            for rank in range(5):
                if rank < len(new_preds) and new_preds[rank].get("field"):
                    p = new_preds[rank]
                    field = p.get("field", "")
                    fid = p.get("field_id", "") or "N/A"
                    score = p.get("score", 0)
                    hdist = p.get("h_distance", -1)
                    
                    is_match = field.lower() == gt_label.lower() or hdist == 0
                    row_class = "match" if is_match else ""
                    dist_class = f"dist-{hdist}" if hdist >= 0 else "dist-na"
                    match_icon = '<span class="match-icon">✓</span>' if is_match else ""
                    hdist_display = str(hdist) if hdist >= 0 else "N/A"
                    
                    html += f'<tr class="{row_class}"><td><strong>{rank+1}</strong></td><td>{field}{match_icon}</td><td class="field-id">{fid}</td><td>{score:.2f}</td><td class="{dist_class}">{hdist_display}</td></tr>'
                else:
                    html += f'<tr><td>{rank+1}</td><td colspan="4" class="no-data">No prediction</td></tr>'
            
            html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
"""
        
        # Legend
        html += """
    <h2>📖 Legend & Scoring Explanation</h2>
    <div class="legend-box">
        <h4>ORKG Scoring System</h4>
        <p>The ORKG accuracy score is a <strong>position-weighted metric</strong> that rewards correct predictions appearing higher in the ranked list:</p>
        <ul>
            <li><strong>Rank 1 (Top-1 Match)</strong>: 100% score - The correct field is the top prediction</li>
            <li><strong>Rank 2 or 3</strong>: 80% score - Correct field appears in positions 2-3</li>
            <li><strong>Rank 4 or 5</strong>: 60% score - Correct field appears in positions 4-5</li>
            <li><strong>Not in Top-5</strong>: 0% score - Correct field not found in top 5 predictions</li>
        </ul>
        <p><em>Final score = Average of all paper scores</em></p>
        
        <h4 style="margin-top:25px;">Hierarchical Distance (H-Distance)</h4>
        <p>Measures how far apart two fields are in the ORKG taxonomy tree:</p>
        <ul>
            <li><span class="dist-0">0</span> = <strong>Exact match</strong> - Same field in taxonomy</li>
            <li><span class="dist-1">1-2</span> = <strong>Close</strong> - Sibling or near-sibling fields (same parent)</li>
            <li><span class="dist-3">3+</span> = <strong>Distant</strong> - Different branches of taxonomy tree</li>
            <li><span class="dist-na">N/A</span> = Field not found in taxonomy</li>
        </ul>
        <p><em>Note: H-Distance is for visualization only and does not affect accuracy scores.</em></p>
        
        <h4 style="margin-top:25px;">Match Types</h4>
        <ul>
            <li><strong>Top-1 Match</strong>: The first prediction exactly matches the ground truth</li>
            <li><strong>Top-5 Match</strong>: The ground truth appears somewhere in the top 5 predictions</li>
        </ul>
    </div>
</div>
</body>
</html>
"""
        return html
    
    def generate_combined_report(self) -> str:
        """Generate combined comparison report."""
        if not self.all_results:
            return ""
        
        approaches = list(self.all_results.keys())
        
        # Generate all charts
        combined_chart = self.create_combined_chart()
        radar_chart = self.create_radar_chart()
        rank_dist_chart = self.create_combined_rank_distribution()
        time_per_paper_chart = self.create_response_time_per_paper()
        timing_breakdown_chart = self.create_timing_breakdown_chart()
        token_usage_chart = self.create_token_usage_chart()
        hdistance_chart = self.create_hdistance_comparison_chart()
        
        # Get old metrics from first approach
        first = list(self.all_results.values())[0]
        old_m = first.get("old_metrics", {})
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Research Field Classifier - Combined Evaluation Report</title>
    <meta charset="UTF-8">
    {self.get_css()}
</head>
<body>
<div class="container">
    <h1>📊 Research Field Classifier - Combined Evaluation Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Approaches Compared:</strong> Old (Flat), {', '.join([a.replace('_', ' ').title() for a in approaches])}</p>
    
    <h2>Summary Comparison Table</h2>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Approach</th>
                <th>Accuracy Score</th>
                <th>Top-1 Accuracy</th>
                <th>Top-5 Accuracy</th>
                <th>Avg H-Distance</th>
                <th>Avg Response Time</th>
                <th>Total Tokens</th>
            </tr>
        </thead>
        <tbody>
            <tr class="old-row">
                <td><strong>🔴 Old (Flat)</strong></td>
                <td>{old_m.get('accuracy_score', 0)*100:.1f}%</td>
                <td>{old_m.get('top1_accuracy', 0)*100:.1f}%</td>
                <td>{old_m.get('top5_accuracy', 0)*100:.1f}%</td>
                <td>{old_m.get('avg_h_distance', 0):.2f}</td>
                <td>{old_m.get('avg_response_time_ms', 0):.0f} ms</td>
                <td>N/A</td>
            </tr>
"""
        
        # Find best values
        all_metrics = [old_m] + [self.all_results[a].get("metrics", {}) for a in approaches]
        best_top1 = max(m.get("top1_accuracy", 0) for m in all_metrics)
        best_top5 = max(m.get("top5_accuracy", 0) for m in all_metrics)
        best_acc = max(m.get("accuracy_score", 0) for m in all_metrics)
        
        for approach in approaches:
            m = self.all_results[approach].get("metrics", {})
            is_best = m.get("accuracy_score", 0) == best_acc and best_acc > 0
            row_class = "best" if is_best else ""
            
            html += f"""
            <tr class="{row_class}">
                <td><strong>🟢 {approach.replace("_", " ").title()}</strong></td>
                <td>{m.get('accuracy_score', 0)*100:.1f}%</td>
                <td>{m.get('top1_accuracy', 0)*100:.1f}%</td>
                <td>{m.get('top5_accuracy', 0)*100:.1f}%</td>
                <td>{m.get('avg_h_distance', 0):.2f}</td>
                <td>{m.get('avg_response_time_ms', 0):.0f} ms</td>
                <td>{m.get('total_tokens', 0):,}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
"""
        
        # Radar Chart - Multi-dimensional comparison
        if radar_chart:
            html += f"""
    <h2>🎯 Multi-Dimensional Comparison (Radar Chart)</h2>
    <p>This radar chart shows how each approach performs across multiple dimensions. Higher values (further from center) are better for all metrics.</p>
    <div class="chart-box full">
        <img src="data:image/png;base64,{radar_chart}">
    </div>
"""
        
        # Combined comparison chart
        if combined_chart:
            html += f"""
    <h2>📈 Accuracy & Performance Comparison</h2>
    <div class="chart-box full">
        <img src="data:image/png;base64,{combined_chart}">
    </div>
"""
        
        # Rank Distribution
        if rank_dist_chart:
            html += f"""
    <h2>📊 Rank Distribution Comparison</h2>
    <p>Shows where ground truth appears in the predictions for each approach. More papers at Rank 1 = better accuracy.</p>
    <div class="chart-box full">
        <img src="data:image/png;base64,{rank_dist_chart}">
    </div>
"""
        
        # Response Time per Paper
        if time_per_paper_chart:
            html += f"""
    <h2>⏱️ Response Time per Paper</h2>
    <p>Response time comparison for each individual paper across all approaches.</p>
    <div class="chart-box full">
        <img src="data:image/png;base64,{time_per_paper_chart}">
    </div>
"""
        
        # Timing Breakdown
        if timing_breakdown_chart:
            html += f"""
    <h2>⚙️ Timing Breakdown</h2>
    <p>Breakdown of time spent in LLM processing vs other operations (network, parsing, etc.)</p>
    <div class="chart-box full">
        <img src="data:image/png;base64,{timing_breakdown_chart}">
    </div>
"""
        
        # Token Usage
        if token_usage_chart:
            html += f"""
    <h2>💰 Token Usage Analysis</h2>
    <p>LLM token consumption comparison - important for cost estimation.</p>
    <div class="chart-box full">
        <img src="data:image/png;base64,{token_usage_chart}">
    </div>
"""
        
        # H-Distance Comparison
        if hdistance_chart:
            html += f"""
    <h2>🌳 Hierarchical Distance Analysis</h2>
    <p>H-Distance measures how far predictions are from ground truth in the taxonomy tree. Lower is better.</p>
    <div class="chart-box full">
        <img src="data:image/png;base64,{hdistance_chart}">
    </div>
"""
        
        html += """
    <h2>📖 Legend & Scoring Explanation</h2>
    <div class="legend-box">
        <h4>ORKG Scoring System</h4>
        <p>The ORKG accuracy score is a <strong>position-weighted metric</strong> that rewards correct predictions appearing higher in the ranked list:</p>
        <ul>
            <li><strong>Rank 1 (Top-1 Match)</strong>: 100% score - The correct field is the top prediction</li>
            <li><strong>Rank 2 or 3</strong>: 80% score - Correct field appears in positions 2-3</li>
            <li><strong>Rank 4 or 5</strong>: 60% score - Correct field appears in positions 4-5</li>
            <li><strong>Not in Top-5</strong>: 0% score - Correct field not found in top 5 predictions</li>
        </ul>
        <p><em>Final score = Average of all paper scores. Higher is better.</em></p>
        
        <h4 style="margin-top:25px;">Metrics Explained</h4>
        <ul>
            <li><strong>Accuracy Score</strong>: ORKG position-weighted accuracy (see above)</li>
            <li><strong>Top-1 Accuracy</strong>: Percentage of papers where the first prediction is correct</li>
            <li><strong>Top-5 Accuracy</strong>: Percentage of papers where correct field appears in top 5</li>
            <li><strong>Avg H-Distance</strong>: Average hierarchical distance of top-1 predictions (lower is better)</li>
            <li><strong>Avg Response Time</strong>: Average time to get classification (lower is better)</li>
            <li><strong>Total Tokens</strong>: Total LLM tokens used (only for new classifiers)</li>
        </ul>
        
        <h4 style="margin-top:25px;">Radar Chart Dimensions</h4>
        <ul>
            <li><strong>Top-1/5 Accuracy</strong>: Direct accuracy percentage (0-100%)</li>
            <li><strong>ORKG Score</strong>: Position-weighted accuracy (0-100%)</li>
            <li><strong>Speed</strong>: Inverted response time (100 = fastest)</li>
            <li><strong>Token Efficiency</strong>: Inverted token usage (100 = fewest tokens)</li>
            <li><strong>Proximity (H-Dist)</strong>: Inverted H-Distance (100 = closest to ground truth)</li>
        </ul>
        
        <h4 style="margin-top:25px;">Hierarchical Distance</h4>
        <ul>
            <li><span class="dist-0">0</span> = Exact match in taxonomy</li>
            <li><span class="dist-1">1-2</span> = Close (sibling fields)</li>
            <li><span class="dist-3">3+</span> = Distant (different branches)</li>
        </ul>
    </div>
</div>
</body>
</html>
"""
        return html
    
    def save_reports(self):
        """Save all reports."""
        for approach in self.all_results:
            output_dir = self.reports_dir / approach
            output_dir.mkdir(parents=True, exist_ok=True)
            
            html = self.generate_approach_report(approach)
            path = output_dir / f"evaluation_report_{self.timestamp}.html"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Saved {approach} report to {path}")
        
        # Combined
        output_dir = self.reports_dir / "combined"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        html = self.generate_combined_report()
        path = output_dir / f"evaluation_report_{self.timestamp}.html"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Saved combined report to {path}")
    
    def run(self):
        """Run visualization."""
        self.load_results()
        
        if not self.all_results:
            logger.error("No results found. Run evaluation_script.py first.")
            return
        
        self.save_reports()
        logger.info(f"\nVisualization complete! Reports saved to: {self.reports_dir}")


def main():
    results_dir = Path(settings.EVALUATION_RESULTS_DIR)
    reports_dir = Path(settings.REPORTS_DIR)
    
    vis = Visualizer(results_dir, reports_dir)
    vis.run()


if __name__ == "__main__":
    main()
