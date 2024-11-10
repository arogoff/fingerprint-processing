import os
import numpy as np
from skimage import io, filters, morphology, feature
from skimage.morphology import skeletonize
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import cv2

class HybridAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        print(f"\n[{self._get_timestamp()}] Initializing Hybrid Analyzer...")
        print(f"[{self._get_timestamp()}] Base path set to: {self.base_path}")
        
        # Initialize weights for hybrid decision
        self.weights = {
            'bifurcation': 0.4,
            'crossover': 0.3,
            'dot_island': 0.3
        }
        
    def _get_timestamp(self):
        """Get current timestamp for logging."""
        return datetime.now().strftime("%H:%M:%S")
        
    def load_and_preprocess(self, image_path, method='bifurcation'):
        """Load and preprocess fingerprint image based on method."""
        try:
            img = io.imread(image_path, as_gray=True)
            
            if method == 'bifurcation':
                # Bifurcation preprocessing
                img = (img - np.min(img)) / (np.max(img) - np.min(img))
                img = (img * 255).astype(np.uint8)
                binary = cv2.adaptiveThreshold(
                    img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
                )
                binary = morphology.remove_small_objects(binary.astype(bool), min_size=50)
                binary = morphology.remove_small_holes(binary, area_threshold=50)
                
            elif method == 'crossover':
                # Crossover preprocessing
                img_blur = filters.gaussian(img, sigma=1)
                thresh = filters.threshold_otsu(img_blur)
                binary = img_blur > thresh
                binary = skeletonize(binary)
                
            else:  # dot_island
                # Dot/Island preprocessing
                img_blur = filters.gaussian(img, sigma=1)
                thresh = filters.threshold_otsu(img_blur)
                binary = img_blur > thresh
                
            return binary
            
        except Exception as e:
            print(f"[{self._get_timestamp()}] Error processing image {image_path}: {str(e)}")
            raise

    def analyze_pair(self, f_image_path, s_image_path):
        """Analyze a fingerprint pair using all three methods."""
        try:
            print(f"[{self._get_timestamp()}] Processing image pair:")
            print(f"  → First image: {f_image_path.name}")
            print(f"  → Second image: {s_image_path.name}")
            
            start_time = time.time()
            
            # Get scores from each method
            bifurcation_score = self._get_bifurcation_score(f_image_path, s_image_path)
            crossover_score = self._get_crossover_score(f_image_path, s_image_path)
            dot_island_score = self._get_dot_island_score(f_image_path, s_image_path)
            
            # Calculate weighted hybrid score
            hybrid_score = (
                self.weights['bifurcation'] * bifurcation_score +
                self.weights['crossover'] * crossover_score +
                self.weights['dot_island'] * dot_island_score
            )
            
            processing_time = time.time() - start_time
            print(f"[{self._get_timestamp()}] Analysis complete (Time: {processing_time:.2f}s)")
            print(f"  → Bifurcation score: {bifurcation_score:.4f}")
            print(f"  → Crossover score: {crossover_score:.4f}")
            print(f"  → Dot/Island score: {dot_island_score:.4f}")
            print(f"  → Hybrid score: {hybrid_score:.4f}")
            print("-" * 80)
            
            return {
                'hybrid': hybrid_score,
                'bifurcation': bifurcation_score,
                'crossover': crossover_score,
                'dot_island': dot_island_score
            }
            
        except Exception as e:
            print(f"[{self._get_timestamp()}] Error in hybrid analysis: {str(e)}")
            raise

    def _get_bifurcation_score(self, f_image_path, s_image_path):
        """Get similarity score using bifurcation method."""
        # Load and preprocess images
        f_binary = self.load_and_preprocess(f_image_path, 'bifurcation')
        s_binary = self.load_and_preprocess(s_image_path, 'bifurcation')
        
        # Extract features
        f_features = self._extract_bifurcation_features(f_binary)
        s_features = self._extract_bifurcation_features(s_binary)
        
        # Compare features
        return self._compare_bifurcation_features(f_features, s_features)

    def _get_crossover_score(self, f_image_path, s_image_path):
        """Get similarity score using crossover method."""
        # Load and preprocess images
        f_binary = self.load_and_preprocess(f_image_path, 'crossover')
        s_binary = self.load_and_preprocess(s_image_path, 'crossover')
        
        # Extract features
        f_features = self._extract_crossover_features(f_binary)
        s_features = self._extract_crossover_features(s_binary)
        
        # Compare features
        return self._compare_crossover_features(f_features, s_features)

    def _get_dot_island_score(self, f_image_path, s_image_path):
        """Get similarity score using dot/island method."""
        # Load and preprocess images
        f_binary = self.load_and_preprocess(f_image_path, 'dot_island')
        s_binary = self.load_and_preprocess(s_image_path, 'dot_island')
        
        # Extract features
        f_features = self._extract_dot_island_features(f_binary)
        s_features = self._extract_dot_island_features(s_binary)
        
        # Compare features
        return self._compare_dot_island_features(f_features, s_features)

    def _extract_bifurcation_features(self, binary_image):
            """Extract bifurcation points from the fingerprint image"""
            skeleton = skeletonize(binary_image)
            bifurcation_points = self._detect_bifurcations(skeleton)
            
            return {
                'skeleton': skeleton,
                'bifurcation_points': bifurcation_points,
                'num_points': len(bifurcation_points)
            }
    
    def _extract_crossover_features(self, skeleton):
        """Extract crossover points from the skeletonized image."""
        rows, cols = skeleton.shape
        crossovers = []
        crossover_map = np.zeros_like(skeleton, dtype=bool)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if skeleton[i, j]:
                    neighborhood = skeleton[i-1:i+2, j-1:j+2]
                    neighbor_count = np.sum(neighborhood) - 1
                    if neighbor_count >= 4:
                        crossovers.append((i, j))
                        crossover_map[i, j] = True
        
        total_pixels = skeleton.size
        crossover_density = len(crossovers) / total_pixels
        
        crossover_distances = []
        if len(crossovers) > 1:
            crossover_points = np.array(crossovers)
            for i in range(len(crossovers)):
                for j in range(i+1, len(crossovers)):
                    dist = np.linalg.norm(crossover_points[i] - crossover_points[j])
                    crossover_distances.append(dist)
        
        return {
            'crossover_count': len(crossovers),
            'crossover_locations': np.array(crossovers),
            'crossover_density': crossover_density,
            'crossover_distances': np.array(crossover_distances),
            'mean_distance': np.mean(crossover_distances) if crossover_distances else 0,
            'std_distance': np.std(crossover_distances) if crossover_distances else 0
        }
    
    def _extract_dot_island_features(self, binary_image):
        """Extract dot and island features from binary image."""
        dots = morphology.remove_small_objects(binary_image, min_size=5, connectivity=2)
        dots = morphology.remove_small_holes(dots, area_threshold=5)
        
        blobs = feature.blob_log(
            binary_image.astype(float),
            min_sigma=1,
            max_sigma=4,
            threshold=0.1
        )
        
        total_white_pixels = np.sum(binary_image)
        image_area = binary_image.size
        density = total_white_pixels / image_area
        
        return {
            'dot_count': np.sum(dots),
            'island_locations': blobs[:, :2] if len(blobs) > 0 else np.array([]),
            'island_count': len(blobs),
            'density': density,
            'total_features': np.sum(dots) + len(blobs)
        }
    
    def _detect_bifurcations(self, skeleton):
        """Detect bifurcation points in the skeleton image"""
        kernel = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=np.float64)
        
        bifurcation_points = []
        
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j]:
                    neighborhood = skeleton[i-1:i+2, j-1:j+2].astype(np.float64)
                    neighbor_count = np.sum(neighborhood * kernel, dtype=np.float64) - 1.0
                    if neighbor_count > 2:
                        bifurcation_points.append((i, j))
        
        return bifurcation_points
    
    def _compare_bifurcation_features(self, features1, features2, threshold: float = 20.0):
        """Compare two sets of bifurcation features."""
        points1 = features1['bifurcation_points']
        points2 = features2['bifurcation_points']
        
        if not points1 or not points2:
            return 0.0
        
        matches = 0
        for p1 in points1:
            min_dist = float('inf')
            for p2 in points2:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_dist = min(min_dist, dist)
            if min_dist < threshold:
                matches += 1
        
        return (2.0 * matches) / (len(points1) + len(points2))
    
    def _compare_crossover_features(self, features1, features2):
        """Compare two sets of crossover features."""
        count_diff = abs(features1['crossover_count'] - features2['crossover_count'])
        count_score = 1 - self._normalize_value(count_diff, 0, 50)
        
        density_diff = abs(features1['crossover_density'] - features2['crossover_density'])
        density_score = 1 - density_diff
        
        distance_diff = abs(features1['mean_distance'] - features2['mean_distance'])
        distance_score = 1 - self._normalize_value(distance_diff, 0, 20)
        
        weights = {'count': 0.4, 'density': 0.3, 'distance': 0.3}
        return (
            weights['count'] * count_score +
            weights['density'] * density_score +
            weights['distance'] * distance_score
        )
    
    def _compare_dot_island_features(self, features1, features2):
        """Compare two sets of dot/island features."""
        dot_diff = abs(features1['dot_count'] - features2['dot_count'])
        dot_score = 1 - self._normalize_value(dot_diff, 0, 100)
        
        island_count_diff = abs(features1['island_count'] - features2['island_count'])
        island_score = 1 - self._normalize_value(island_count_diff, 0, 20)
        
        density_diff = abs(features1['density'] - features2['density'])
        density_score = 1 - density_diff
        
        weights = {'dot': 0.4, 'island': 0.4, 'density': 0.2}
        return (
            weights['dot'] * dot_score +
            weights['island'] * island_score +
            weights['density'] * density_score
        )
    
    def _normalize_value(self, value, min_val=0, max_val=1000):
        """Normalize a value to [0,1] range."""
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def run_analysis(self, start_idx=1501, end_idx=2000):
        """Run the complete analysis on the test set."""
        print(f"\n[{self._get_timestamp()}] Starting analysis on test set (images {start_idx}-{end_idx})")
        
        genuine_pairs = []
        genuine_scores = []
        total_pairs = end_idx - start_idx + 1
        processed_pairs = 0
        
        analysis_start_time = time.time()
        
        # Process genuine pairs
        for idx in range(start_idx, end_idx + 1):
            pair_start_time = time.time()
            
            f_pattern = f'f{idx:04d}'
            s_pattern = f's{idx:04d}'
            
            try:
                f_image = self.find_image_file(f_pattern)
                s_image = self.find_image_file(s_pattern)
                
                if f_image and s_image:
                    genuine_pairs.append((f_image, s_image))
                    scores = self.analyze_pair(f_image, s_image)
                    genuine_scores.append(scores)
                    processed_pairs += 1
                    
                    progress = (processed_pairs / total_pairs) * 100
                    print(f"[{self._get_timestamp()}] Progress: {progress:.1f}% complete")
            except Exception as e:
                print(f"[{self._get_timestamp()}] Error processing pair {idx}: {str(e)}")
                continue
        
        # Generate and analyze impostor pairs
        print(f"\n[{self._get_timestamp()}] Generating and analyzing impostor pairs...")
        impostor_pairs = self.generate_impostor_pairs(genuine_pairs, len(genuine_pairs))
        impostor_scores = []
        
        for f_image, s_image in impostor_pairs:
            try:
                scores = self.analyze_pair(f_image, s_image)
                impostor_scores.append(scores)
            except Exception as e:
                print(f"[{self._get_timestamp()}] Error processing impostor pair: {str(e)}")
                continue
        
        # Calculate error rates for each method and hybrid
        methods = ['hybrid', 'bifurcation', 'crossover', 'dot_island']
        results = {}
        
        for method in methods:
            genuine_method_scores = [score[method] for score in genuine_scores]
            impostor_method_scores = [score[method] for score in impostor_scores]
            
            far_stats, frr_stats, eer, eer_threshold = self.calculate_error_rates(
                genuine_method_scores, impostor_method_scores, method
            )
            
            results[method] = {
                'far_stats': far_stats,
                'frr_stats': frr_stats,
                'eer': eer,
                'eer_threshold': eer_threshold
            }
        
        return results
    
    def find_image_file(self, image_name: str) -> Path:
        """Find the full path of an image file in the dataset."""
        direct_path = self.base_path / f"{image_name}.png"
        if direct_path.exists():
            return direct_path
            
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.startswith(image_name) and file.endswith('.png'):
                    return Path(root) / file
                    
        raise FileNotFoundError(f"Could not find image file for {image_name}")
    
    def generate_impostor_pairs(self, genuine_pairs, num_impostor_pairs):
        """Generate impostor pairs by mixing different identities."""
        impostor_pairs = []
        all_f_images = [pair[0] for pair in genuine_pairs]
        all_s_images = [pair[1] for pair in genuine_pairs]
        
        while len(impostor_pairs) < num_impostor_pairs:
            f_idx = random.randint(0, len(all_f_images) - 1)
            s_idx = random.randint(0, len(all_s_images) - 1)
            
            if f_idx != s_idx:
                impostor_pairs.append((all_f_images[f_idx], all_s_images[s_idx]))
        
        return impostor_pairs
    
    def calculate_error_rates(self, genuine_scores, impostor_scores, method_name):
        """Calculate error rates for a specific method."""
        print(f"\n[{self._get_timestamp()}] Calculating error rates for {method_name}...")
        
        # Convert to numpy arrays
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Generate ROC curve
        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        scores = np.concatenate([genuine_scores, impostor_scores])
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        
        # Calculate FAR and FRR
        far_rates = []
        frr_rates = []
        
        for threshold in thresholds:
            frr = np.mean(genuine_scores < threshold)
            far = np.mean(impostor_scores >= threshold)
            
            far_rates.append(far)
            frr_rates.append(frr)
        
        far_rates = np.array(far_rates)
        frr_rates = np.array(frr_rates)
        
        # Find EER
        eer_idx = np.argmin(np.abs(far_rates - frr_rates))
        eer = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
        
        # Calculate statistics
        far_stats = {
            'min': np.min(far_rates),
            'max': np.max(far_rates),
            'avg': np.mean(far_rates)
        }
        
        frr_stats = {
            'min': np.min(frr_rates),
            'max': np.max(frr_rates),
            'avg': np.mean(frr_rates)
        }
        
        # Plot ROC curve
        self.plot_error_rates(fpr, tpr, eer, method_name)
        
        return far_stats, frr_stats, eer, thresholds[eer_idx]
    
    def plot_error_rates(self, fpr, tpr, eer, method_name):
        """Generate and save ROC curve plot."""
        plt.figure(figsize=(10, 8))
        
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f}, EER = {eer:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        
        eer_point = np.argmin(np.abs(1 - tpr - fpr))
        plt.plot(fpr[eer_point], tpr[eer_point], 'ro', 
                label=f'EER Point ({eer:.4f})')
        
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1-FRR)')
        plt.title(f'ROC Curve - {method_name.capitalize()}')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        # Save plot
        output_path = Path(r'C:\Users\GCCISAdmin\Downloads\group1-biometrics\data') / f'roc_curve_{method_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print(f"\n{'='*80}")
    print("NIST Fingerprint Database - Hybrid Analysis")
    print(f"{'='*80}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize analyzer
        base_path = Path(r"C:\Users\GCCISAdmin\Downloads\group1-biometrics\NISTSpecialDatabase4GrayScaleImagesofFIGS")
        analyzer = HybridAnalyzer(base_path)
        
        # Create data directory if it doesn't exist
        data_dir = Path(r"C:\Users\GCCISAdmin\Downloads\group1-biometrics\data")
        data_dir.mkdir(exist_ok=True)
        
        # Run analysis
        results = analyzer.run_analysis(start_idx=1501, end_idx=2000)
        
        # Print results for each method
        methods = ['hybrid', 'bifurcation', 'crossover', 'dot_island']
        
        for method in methods:
            print(f"\nResults for {method.capitalize()} Method:")
            print("=" * 40)
            print(f"False Rejection Rate (FRR):")
            print(f"  → Minimum: {results[method]['frr_stats']['min']:.4f}")
            print(f"  → Maximum: {results[method]['frr_stats']['max']:.4f}")
            print(f"  → Average: {results[method]['frr_stats']['avg']:.4f}")
            print(f"\nFalse Acceptance Rate (FAR):")
            print(f"  → Minimum: {results[method]['far_stats']['min']:.4f}")
            print(f"  → Maximum: {results[method]['far_stats']['max']:.4f}")
            print(f"  → Average: {results[method]['far_stats']['avg']:.4f}")
            print(f"\nEqual Error Rate (EER): {results[method]['eer']:.4f}")
            print(f"EER Threshold: {results[method]['eer_threshold']:.4f}")
        
        # Create methods table for lab report
        print("\nMethods Table:")
        print("-" * 120)
        print("| Method      | FRR avg | FRR min | FRR max | FAR avg | FAR min | FAR max | EER |")
        print("|-------------|---------|---------|---------|---------|---------|---------|-----|")
        
        for method in methods:
            print(f"|{method:12}|{results[method]['frr_stats']['avg']:8.3%} |"
                  f"{results[method]['frr_stats']['min']:8.3%} |"
                  f"{results[method]['frr_stats']['max']:8.3%} |"
                  f"{results[method]['far_stats']['avg']:8.3%} |"
                  f"{results[method]['far_stats']['min']:8.3%} |"
                  f"{results[method]['far_stats']['max']:8.3%} |"
                  f"{results[method]['eer']:4.3%}|")
        
        print("\nROC curves have been saved to the data directory.")
        
    except Exception as e:
        print(f"\nERROR: An error occurred during analysis: {str(e)}")
        raise
    finally:
        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")

if __name__ == "__main__":
    main()