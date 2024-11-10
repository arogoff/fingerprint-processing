import os
import numpy as np
from skimage import io, filters, morphology, feature
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random

class DotIslandAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.threshold = 0.15
        print(f"\n[{self._get_timestamp()}] Initializing Dot/Island Analyzer...")
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
            
            return binary
        except Exception as e:
            print(f"[{self._get_timestamp()}] Error processing image {image_path}: {str(e)}")
            raise
    
    def extract_dots_islands(self, binary_image):
        """Extract dot and island features from binary image."""
        # Find small connected components (dots)
        dots = morphology.remove_small_objects(binary_image, min_size=5, connectivity=2)
        dots = morphology.remove_small_holes(dots, area_threshold=5)
        
        # Use blob detection to find islands
        blobs = feature.blob_log(
            binary_image.astype(float),
            min_sigma=1,
            max_sigma=4,
            threshold=0.1
        )
        
        # Calculate additional features
        total_white_pixels = np.sum(binary_image)
        image_area = binary_image.size
        density = total_white_pixels / image_area
        
        # Combine features into a descriptor
        features = {
            'dot_count': np.sum(dots),
            'island_locations': blobs[:, :2] if len(blobs) > 0 else np.array([]),
            'island_count': len(blobs),
            'density': density,
            'total_features': np.sum(dots) + len(blobs)
        }
        
        return features
    
    def normalize_similarity(self, value, min_val=0, max_val=1000):
        """Normalize similarity scores to [0,1] range."""
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def compare_features(self, features1, features2):
        """Compare two sets of features and return a similarity score."""
        # Compare dot counts (normalized)
        dot_diff = abs(features1['dot_count'] - features2['dot_count'])
        dot_score = self.normalize_similarity(dot_diff, 0, 100)
        
        # Compare island counts (normalized)
        island_count_diff = abs(features1['island_count'] - features2['island_count'])
        island_score = self.normalize_similarity(island_count_diff, 0, 20)
        
        # Compare density
        density_diff = abs(features1['density'] - features2['density'])
        density_score = 1 - density_diff
        
        # Calculate weighted similarity score
        weights = {'dot': 0.4, 'island': 0.4, 'density': 0.2}
        similarity = (
            weights['dot'] * (1 - dot_score) +
            weights['island'] * (1 - island_score) +
            weights['density'] * density_score
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
            
            f_binary = self.load_and_preprocess(f_image_path)
            s_binary = self.load_and_preprocess(s_image_path)
            
            # Extract features
            print(f"[{self._get_timestamp()}] Extracting features...")
            f_features = self.extract_dots_islands(f_binary)
            s_features = self.extract_dots_islands(s_binary)
            
            # Compare features
            similarity = self.compare_features(f_features, s_features)
            
            processing_time = time.time() - start_time
            print(f"[{self._get_timestamp()}] Pair analysis complete (Time: {processing_time:.2f}s)")
            print(f"  → Similarity score: {similarity:.4f}")
            print(f"  → Features found: {f_features['dot_count']} dots, {f_features['island_count']} islands (first image)")
            print(f"  → Features found: {s_features['dot_count']} dots, {s_features['island_count']} islands (second image)")
            print("-" * 80)
            
            return similarity
        except Exception as e:
            print(f"[{self._get_timestamp()}] Error analyzing image pair: {str(e)}")
            raise
    
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
        # Remove any NaN values before calculating statistics
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
        
        print(f"[{self._get_timestamp()}] Error rate calculation complete")
        print(f"[{self._get_timestamp()}] Number of thresholds evaluated: {len(thresholds)}")
        print(f"[{self._get_timestamp()}] FRR range: {frr_stats['min']:.4f} - {frr_stats['max']:.4f}")
        print(f"[{self._get_timestamp()}] FAR range: {far_stats['min']:.4f} - {far_stats['max']:.4f}")
        
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
        
        # Add labels and title
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1-FRR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        
        # Save plot
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
        
        # Collect genuine pairs
        for idx in range(start_idx, end_idx + 1):
            pair_start_time = time.time()
            
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

def main():
    print(f"\n{'='*80}")
    print("NIST Fingerprint Database - Dot/Island Analysis")
    print(f"{'='*80}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize analyzer with path to dataset
        base_path = Path('NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt')
        analyzer = DotIslandAnalyzer(base_path)
        
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