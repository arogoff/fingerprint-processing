import numpy as np
import cv2
from pathlib import Path
from skimage import io, morphology, filters, util
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import pandas as pd
from tqdm import tqdm

class BifurcationAnalyzer:
    def __init__(self, dataset_path: str):
        """
        Initialize the bifurcation analyzer
        Args:
            dataset_path: Path to the extracted NIST dataset folder
        """
        self.dataset_path = Path(dataset_path)
        # Verify the dataset structure
        self.verify_dataset_structure()
        
    def verify_dataset_structure(self):
        """Verify and print the dataset structure"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

    def find_image_file(self, image_name: str) -> Path:
        """
        Find the full path of an image file in the dataset
        Args:
            image_name: Base name of the image file (e.g., 'f0001')
        Returns:
            Path to the image file
        """
        # Search for the image file recursively
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.startswith(image_name) and file.endswith('.png'):
                    return Path(root) / file
        raise FileNotFoundError(f"Could not find image file for {image_name}")

    def load_image(self, image_base_name: str) -> np.ndarray:
        """Load and preprocess a fingerprint image"""
        try:
            # Find the full path to the image
            img_path = self.find_image_file(image_base_name)
            
            # Load image
            img = io.imread(str(img_path), as_gray=True)
            
            # Basic preprocessing
            img = self._preprocess_image(img)
            return img
        except Exception as e:
            print(f"Error loading image {image_base_name}: {str(e)}")
            raise

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess the image for better feature extraction"""
        # Normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        
        # Apply adaptive thresholding
        img = (img * 255).astype(np.uint8)
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        
        # Remove noise
        img = morphology.remove_small_objects(img.astype(bool), min_size=50)
        img = morphology.remove_small_holes(img, area_threshold=50)
        
        return img

    def extract_bifurcation_features(self, img: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Extract bifurcation points from the fingerprint image"""
        # Thin the ridges
        skeleton = skeletonize(img)
        
        # Find bifurcation points
        bifurcation_points = self._detect_bifurcations(skeleton)
        
        return skeleton, bifurcation_points

    def _detect_bifurcations(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """Detect bifurcation points in the skeleton image"""
        kernel = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=np.uint8)
        
        bifurcation_points = []
        
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j]:
                    neighborhood = skeleton[i-1:i+2, j-1:j+2].astype(np.uint8)
                    neighbor_count = np.sum(neighborhood * kernel) - 1
                    if neighbor_count > 2:
                        bifurcation_points.append((i, j))
        
        return bifurcation_points

    def compare_fingerprints(self, points1: List[Tuple[int, int]], 
                           points2: List[Tuple[int, int]], 
                           threshold: float = 20.0) -> float:
        """Compare two sets of bifurcation points and return a similarity score"""
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
        
        score = (2.0 * matches) / (len(points1) + len(points2))
        return score

    def visualize_results(self, img: np.ndarray, 
                         bifurcation_points: List[Tuple[int, int]], 
                         title: str = "Fingerprint Analysis"):
        """Visualize the original image with detected bifurcation points"""
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        
        if bifurcation_points:
            points = np.array(bifurcation_points)
            plt.plot(points[:, 1], points[:, 0], 'r.', markersize=5)
            
        plt.title(title)
        plt.axis('off')
        plt.show()

    def make_match_decision(self, similarity_score):
        """
        Make a match/no-match decision based on the similarity score
        """
        THRESHOLD = 0.6  # Balanced threshold
        return {
            'match': similarity_score >= THRESHOLD,
            'confidence': abs(similarity_score - 0.5) * 2,  # 0 to 1 scale
            'score': similarity_score
        }

if __name__ == "__main__":
    dataset_path = r"C:\Users\GCCISAdmin\Downloads\group1-biometrics\NISTSpecialDatabase4GrayScaleImagesofFIGS"
    analyzer = BifurcationAnalyzer(dataset_path)
    
    # Create lists to store results
    genuine_results = []
    impostor_results = []
    
    # Process genuine pairs (1-1500)
    print("Processing genuine pairs (1-1500)...")
    for i in tqdm(range(1, 1501), desc="Processing genuine pairs"):
        try:
            f_name = f"f{i:04d}"
            s_name = f"s{i:04d}"
            
            img1 = analyzer.load_image(f_name)
            img2 = analyzer.load_image(s_name)
            
            skeleton1, points1 = analyzer.extract_bifurcation_features(img1)
            skeleton2, points2 = analyzer.extract_bifurcation_features(img2)
            
            similarity = analyzer.compare_fingerprints(points1, points2)
            
            genuine_results.append({
                'pair_id': i,
                'f_image': f_name,
                's_image': s_name,
                'similarity_score': similarity,
                'pair_type': 'genuine'
            })
            
        except Exception as e:
            print(f"\nError processing genuine pair {i}: {str(e)}")
    
    # Process impostor pairs (1501-2000 with random matches)
    print("\nProcessing impostor pairs (1501-2000)...")
    np.random.seed(42)  # For reproducibility
    
    for i in tqdm(range(1501, 2001), desc="Processing impostor pairs"):
        try:
            f_name = f"f{i:04d}"
            
            # Choose a random s_image from 1-2000, but not the matching one
            while True:
                random_s = np.random.randint(1, 2001)
                if random_s != i:  # Ensure we don't accidentally pick the matching pair
                    break
                    
            s_name = f"s{random_s:04d}"
            
            img1 = analyzer.load_image(f_name)
            img2 = analyzer.load_image(s_name)
            
            skeleton1, points1 = analyzer.extract_bifurcation_features(img1)
            skeleton2, points2 = analyzer.extract_bifurcation_features(img2)
            
            similarity = analyzer.compare_fingerprints(points1, points2)
            
            impostor_results.append({
                'pair_id': i,
                'f_image': f_name,
                's_image': s_name,
                'similarity_score': similarity,
                'pair_type': 'impostor'
            })
            
        except Exception as e:
            print(f"\nError processing impostor pair {i}: {str(e)}")
    
    # Combine results and save to CSV
    all_results = genuine_results + impostor_results
    df_results = pd.DataFrame(all_results)
    output_dir = Path(r"C:\Users\GCCISAdmin\Downloads\group1-biometrics\data")
    df_results.to_csv(output_dir / 'fingerprint_analysis_results.csv', index=False)
    print("\nResults saved to 'fingerprint_analysis_results.csv'")
    
    # Calculate error rates
    genuine_scores = df_results[df_results['pair_type'] == 'genuine']['similarity_score'].dropna().values
    impostor_scores = df_results[df_results['pair_type'] == 'impostor']['similarity_score'].dropna().values
    
    thresholds = np.linspace(0, 1.2, 1000)
    fars = []
    frrs = []
    
    for threshold in thresholds:
        # FRR: proportion of genuine pairs incorrectly rejected
        frr = np.mean(genuine_scores < threshold)
        # FAR: proportion of impostor pairs incorrectly accepted
        far = np.mean(impostor_scores >= threshold)
        
        fars.append(far)
        frrs.append(frr)
    
    fars = np.array(fars)
    frrs = np.array(frrs)
    
    # Find EER
    eer_idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_idx] + frrs[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    # Print results
    print("\nError Rate Analysis:")
    print("-" * 50)
    print("\nFalse Accept Rate (FAR):")
    print(f"  Minimum: {np.min(fars):.3%}")
    print(f"  Maximum: {np.max(fars):.3%}")
    print(f"  Average: {np.mean(fars):.3%}")
    print("\nFalse Reject Rate (FRR):")
    print(f"  Minimum: {np.min(frrs):.3%}")
    print(f"  Maximum: {np.max(frrs):.3%}")
    print(f"  Average: {np.mean(frrs):.3%}")
    print("\nEqual Error Rate (EER):")
    print(f"  EER: {eer:.3%}")
    print(f"  at threshold: {eer_threshold:.3f}")
    
    # Additional statistics
    print("\nMatching Statistics:")
    print("-" * 50)
    print("Genuine Pairs:")
    print(f"  Average Score: {np.mean(genuine_scores):.3f}")
    print(f"  Min Score: {np.min(genuine_scores):.3f}")
    print(f"  Max Score: {np.max(genuine_scores):.3f}")
    print("\nImpostor Pairs:")
    print(f"  Average Score: {np.mean(impostor_scores):.3f}")
    print(f"  Min Score: {np.min(impostor_scores):.3f}")
    print(f"  Max Score: {np.max(impostor_scores):.3f}")
    
    # Plot distributions and error rates
    plt.figure(figsize=(15, 5))
    
    # Plot score distributions
    plt.subplot(121)
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine Pairs', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor Pairs', density=True)
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    
    # Plot FAR vs FRR
    plt.subplot(122)
    plt.plot(thresholds, fars, label='FAR')
    plt.plot(thresholds, frrs, label='FRR')
    plt.axvline(x=eer_threshold, color='r', linestyle='--', 
               label=f'EER Threshold = {eer_threshold:.3f}')
    plt.axhline(y=eer, color='r', linestyle='--', 
               label=f'EER = {eer:.3%}')
    plt.xlabel('Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR vs FRR Curves')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rates.png')
    plt.show()
    
    # Create a row for your methods table
    print("\nMethods Table Row:")
    print("-" * 50)
    print("| Method | FRR avg | FRR min | FRR max | FAR avg | FAR min | FAR max | EER |")
    print("|---------|----------|----------|----------|----------|----------|----------|-----|")
    print(f"|Bifurcation|{np.mean(frrs):.3%}|{np.min(frrs):.3%}|{np.max(frrs):.3%}|{np.mean(fars):.3%}|{np.min(fars):.3%}|{np.max(fars):.3%}|{eer:.3%}|")