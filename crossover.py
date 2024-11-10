import os
import numpy as np
from skimage import io, filters, morphology, feature, util
from skimage.morphology import skeletonize
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random

class CrossoverAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.threshold = 0.15
        print(f"\n[{self._get_timestamp()}] Initializing Crossover Analyzer...")
        print(f"[{self._get_timestamp()}] Base path set to: {self.base_path}")
        
    def _get_timestamp(self):
        """Get current timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
        
    def load_and_preprocess(self, image_path):
        """Load and preprocess fingerprint image."""
        try:
            # Read image
            img = io.imread(image_path, as_gray=True)
            
            # Apply Gaussian blur to reduce noise
            img_blur = filters.gaussian(img, sigma=1)
            
            # Apply threshold to create binary image
            thresh = filters.threshold_otsu(img_blur)
            binary = img_blur > thresh
            
            # Skeletonize the image to get one-pixel wide ridges
            skeleton = skeletonize(binary)
            
            return skeleton
        except Exception as e:
            print(f"[{self._get_timestamp()}] Error processing image {image_path}: {str(e)}")
            raise
    
    def detect_crossovers(self, skeleton):
        """Detect crossover points in the skeletonized image."""
        # Create 3x3 neighborhood kernel for each pixel
        rows, cols = skeleton.shape
        crossovers = []
        crossover_map = np.zeros_like(skeleton, dtype=bool)
        
        # Scan the skeleton for crossover points
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if skeleton[i, j]:
                    # Get 3x3 neighborhood
                    neighborhood = skeleton[i-1:i+2, j-1:j+2]
                    # Count number of neighbors
                    neighbor_count = np.sum(neighborhood) - 1  # Subtract center pixel
                    
                    # Crossover points typically have 4 neighbors
                    if neighbor_count >= 4:
                        crossovers.append((i, j))
                        crossover_map[i, j] = True
        
        # Calculate additional features
        total_pixels = skeleton.size
        crossover_density = len(crossovers) / total_pixels
        
        # Find distances between crossovers for pattern analysis
        crossover_distances = []
        if len(crossovers) > 1:
            crossover_points = np.array(crossovers)
            for i in range(len(crossovers)):
                for j in range(i+1, len(crossovers)):
                    dist = np.linalg.norm(crossover_points[i] - crossover_points[j])
                    crossover_distances.append(dist)
        
        features = {
            'crossover_count': len(crossovers),
            'crossover_locations': np.array(crossovers),
            'crossover_density': crossover_density,
            'crossover_distances': np.array(crossover_distances),
            'mean_distance': np.mean(crossover_distances) if crossover_distances else 0,
            'std_distance': np.std(crossover_distances) if crossover_distances else 0
        }
        
        return features, crossover_map
    
    def normalize_similarity(self, value, min_val=0, max_val=1000):
        """Normalize similarity scores to [0,1] range."""
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def compare_features(self, features1, features2):
        """Compare two sets of crossover features and return a similarity score."""
        # Compare crossover counts (normalized)
        count_diff = abs(features1['crossover_count'] - features2['crossover_count'])
        count_score = self.normalize_similarity(count_diff, 0, 50)
        
        # Compare density
        density_diff = abs(features1['crossover_density'] - features2['crossover_density'])
        density_score = 1 - density_diff
        
        # Compare mean distances between crossovers
        distance_diff = abs(features1['mean_distance'] - features2['mean_distance'])
        distance_score = self.normalize_similarity(distance_diff, 0, 20)
        
        # Compare standard deviations of distances
        std_diff = abs(features1['std_distance'] - features2['std_distance'])
        std_score = self.normalize_similarity(std_diff, 0, 10)
        
        # Calculate weighted similarity score
        weights = {
            'count': 0.4,
            'density': 0.2,
            'distance': 0.2,
            'std': 0.2
        }
        
        similarity = (
            weights['count'] * (1 - count_score) +
            weights['density'] * density_score +
            weights['distance'] * (1 - distance_score) +
            weights['std'] * (1 - std_score)
        )
        
        return similarity
    
    def analyze_image_pair(self, f_image_path, s_image_path):
        """Analyze a pair of fingerprint images and return similarity score."""
        try:
            # Load and process both images
            print(f"[{self._get_timestamp()}] Processing image pair:")
            print(f"  → First image: {f_image_path.name}")
            print(f"  → Second image: {s_image_path.name}")
            
            start_time = time.time()
            
            f_skeleton = self.load_and_preprocess(f_image_path)
            s_skeleton = self.load_and_preprocess(s_image_path)
            
            # Extract features
            print(f"[{self._get_timestamp()}] Extracting crossover features...")
            f_features, f_crossover_map = self.detect_crossovers(f_skeleton)
            s_features, s_crossover_map = self.detect_crossovers(s_skeleton)
            
            # Compare features
            similarity = self.compare_features(f_features, s_features)
            
            processing_time = time.time() - start_time
            print(f"[{self._get_timestamp()}] Pair analysis complete (Time: {processing_time:.2f}s)")
            print(f"  → Similarity score: {similarity:.4f}")
            print(f"  → Crossovers found: {f_features['crossover_count']} (first image)")
            print(f"  → Crossovers found: {s_features['crossover_count']} (second image)")
            print(f"  → Mean distance between crossovers: {f_features['mean_distance']:.2f} (first image)")
            print(f"  → Mean distance between crossovers: {s_features['mean_distance']:.2f} (second image)")
            print("-" * 80)
            
            # Optionally save visualization of crossover points
            #self.save_visualization(f_image_path, f_skeleton, f_crossover_map, 'first')
            #self.save_visualization(s_image_path, s_skeleton, s_crossover_map, 'second')
            
            return similarity
        except Exception as e:
            print(f"[{self._get_timestamp()}] Error analyzing image pair: {str(e)}")
            raise
    
    def save_visualization(self, image_path, skeleton, crossover_map, suffix):
        """Save visualization of detected crossover points."""
        plt.figure(figsize=(10, 5))
        
        # Original skeleton
        plt.subplot(121)
        plt.imshow(skeleton, cmap='gray')
        plt.title('Skeletonized Image')
        
        # Skeleton with crossover points highlighted
        plt.subplot(122)
        overlay = np.zeros((*skeleton.shape, 3))
        overlay[skeleton] = [0, 0, 1]  # Blue for skeleton
        overlay[crossover_map] = [1, 0, 0]  # Red for crossover points
        plt.imshow(overlay)
        plt.title('Detected Crossovers')
        
        # Save and close
        output_path = f'./crossovers/crossovers_{suffix}_{Path(image_path).stem}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    # [Previous import statements and class methods remain the same up until save_visualization]

    def generate_impostor_pairs(self, genuine_pairs, num_impostor_pairs):
        """Generate impostor pairs by mixing different identities."""
        impostor_pairs = []
        all_f_images = [pair[0] for pair in genuine_pairs]
        all_s_images = [pair[1] for pair in genuine_pairs]
        
        while len(impostor_pairs) < num_impostor_pairs:
            f_idx = random.randint(0, len(all_f_images) - 1)
            s_idx = random.randint(0, len(all_s_images) - 1)
            
            # Ensure we're not accidentally creating a genuine pair
            if f_idx != s_idx:
                impostor_pairs.append((all_f_images[f_idx], all_s_images[s_idx]))
        
        return impostor_pairs

    def calculate_error_rates(self, genuine_scores, impostor_scores):
        """Calculate FRR, FAR, and EER using both genuine and impostor scores."""
        print(f"\n[{self._get_timestamp()}] Calculating error rates...")
        
        # Create labels (1 for genuine, 0 for impostor)
        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        scores = np.concatenate([genuine_scores, impostor_scores])
        
        # Calculate ROC curve and error rates
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        
        # Calculate FNR (False Negative Rate = FRR)
        fnr = 1 - tpr
        
        # Calculate actual FAR and FRR rates at each threshold
        far_rates = []
        frr_rates = []
        
        for threshold in thresholds:
            # Calculate FRR (proportion of genuine pairs incorrectly rejected)
            frr = np.mean(np.array(genuine_scores) < threshold)
            frr_rates.append(frr)
            
            # Calculate FAR (proportion of impostor pairs incorrectly accepted)
            far = np.mean(np.array(impostor_scores) >= threshold)
            far_rates.append(far)
        
        far_rates = np.array(far_rates)
        frr_rates = np.array(frr_rates)
        
        # Find EER
        eer_idx = np.argmin(np.abs(far_rates - frr_rates))
        eer = np.mean([far_rates[eer_idx], frr_rates[eer_idx]])
        
        # Calculate actual min/max/avg statistics
        far_rates_clean = far_rates[~np.isnan(far_rates)]
        frr_rates_clean = frr_rates[~np.isnan(frr_rates)]
        
        far_stats = {
            'min': np.min(far_rates_clean),
            'max': np.max(far_rates_clean),
            'avg': np.mean(far_rates_clean)
        }
        
        frr_stats = {
            'min': np.min(frr_rates_clean),
            'max': np.max(frr_rates_clean),
            'avg': np.mean(frr_rates_clean)
        }
        
        # Generate ROC curve plot
        self.plot_error_rates(fpr, tpr, eer)
        
        return far_stats, frr_stats, eer, thresholds[eer_idx]

    def plot_error_rates(self, fpr, tpr, eer):
        """Generate and save ROC curve plot with additional metrics."""
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f}, EER = {eer:.4f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        
        # Plot EER point
        eer_point = np.argmin(np.abs(1 - tpr - fpr))
        plt.plot(fpr[eer_point], tpr[eer_point], 'ro', 
                label=f'EER Point ({eer:.4f})')
        
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1-FRR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_analysis(self, start_idx=1501, end_idx=2000):
        """Run the complete analysis on the test set."""
        print(f"\n[{self._get_timestamp()}] Starting analysis on test set (images {start_idx}-{end_idx})")
        print(f"[{self._get_timestamp()}] Scanning directories for image pairs...")
        
        genuine_pairs = []
        genuine_scores = []
        total_pairs = end_idx - start_idx + 1
        processed_pairs = 0
        
        analysis_start_time = time.time()
        
        # Collect and analyze genuine pairs
        for idx in range(start_idx, end_idx + 1):
            # Find corresponding files
            f_pattern = f'f{idx:04d}'
            s_pattern = f's{idx:04d}'
            
            print(f"\n[{self._get_timestamp()}] Looking for pair {idx} ({f_pattern}, {s_pattern})")
            
            f_image = None
            s_image = None
            
            # Search through all subdirectories
            for subdir in self.base_path.glob('*'):
                if not subdir.is_dir():
                    continue
                    
                for img_path in subdir.glob('*.png'):
                    if f_pattern in img_path.stem:
                        f_image = img_path
                    elif s_pattern in img_path.stem:
                        s_image = img_path
                        
                if f_image and s_image:
                    break
            
            if f_image and s_image:
                genuine_pairs.append((f_image, s_image))
                similarity = self.analyze_image_pair(f_image, s_image)
                genuine_scores.append(similarity)
                processed_pairs += 1
                
                # Calculate and display progress
                progress = (processed_pairs / total_pairs) * 100
                elapsed_time = time.time() - analysis_start_time
                avg_time_per_pair = elapsed_time / processed_pairs
                estimated_remaining = avg_time_per_pair * (total_pairs - processed_pairs)
                
                print(f"[{self._get_timestamp()}] Progress: {progress:.1f}% complete")
                print(f"[{self._get_timestamp()}] Estimated time remaining: {estimated_remaining:.1f} seconds")
            else:
                print(f"[{self._get_timestamp()}] WARNING: Could not find matching pair for index {idx}")
        
        # Generate and analyze impostor pairs
        print(f"\n[{self._get_timestamp()}] Generating and analyzing impostor pairs...")
        impostor_pairs = self.generate_impostor_pairs(genuine_pairs, len(genuine_pairs))
        impostor_scores = []
        
        for f_image, s_image in impostor_pairs:
            similarity = self.analyze_image_pair(f_image, s_image)
            impostor_scores.append(similarity)
        
        # Calculate error rates
        far_stats, frr_stats, eer, eer_threshold = self.calculate_error_rates(
            genuine_scores, impostor_scores
        )
        
        total_time = time.time() - analysis_start_time
        print(f"\n[{self._get_timestamp()}] Analysis complete!")
        print(f"[{self._get_timestamp()}] Total processing time: {total_time:.2f} seconds")
        print(f"[{self._get_timestamp()}] Processed {processed_pairs} genuine pairs and {len(impostor_pairs)} impostor pairs")
        
        return far_stats, frr_stats, eer, eer_threshold

# [Previous main() function and __name__ == "__main__" check remain the same]
    # The rest of the methods (generate_impostor_pairs, calculate_error_rates, 
    # plot_error_rates, and run_analysis) remain the same as in the original code
    
def main():
    print(f"\n{'='*80}")
    print("NIST Fingerprint Database - Crossover Analysis")
    print(f"{'='*80}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize analyzer with path to dataset
        base_path = Path('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt')
        analyzer = CrossoverAnalyzer(base_path)
        
        # Run analysis on test set (last 500 pairs)
        far_stats, frr_stats, eer, eer_threshold = analyzer.run_analysis()
        
        # Print results
        print("\nFINAL RESULTS:")
        print("=" * 40)
        print(f"False Rejection Rate (FRR):")
        print(f"  → Minimum: {frr_stats['min']:.4f}")
        print(f"  → Maximum: {frr_stats['max']:.4f}")
        print(f"  → Average: {frr_stats['avg']:.4f}")
        print(f"\nFalse Acceptance Rate (FAR):")
        print(f"  → Minimum: {far_stats['min']:.4f}")
        print(f"  → Maximum: {far_stats['max']:.4f}")
        print(f"  → Average: {far_stats['avg']:.4f}")
        print(f"\nEqual Error Rate (EER): {eer:.4f}")
        print(f"EER Threshold: {eer_threshold:.4f}")
        print("\nROC curve has been saved as 'roc_curve.png'")
        
    except Exception as e:
        print(f"\nERROR: An error occurred during analysis: {str(e)}")
        raise
    finally:
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()