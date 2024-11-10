import numpy as np
import cv2
from pathlib import Path
from skimage import io, morphology, filters, util
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

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
        
        # Print folder structure for debugging
        print("Dataset structure:")
        for root, dirs, files in os.walk(self.dataset_path):
            level = root.replace(str(self.dataset_path), '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level == 0:  # Only print files at root level
                subindent = ' ' * 4 * (level + 1)
                for f in files[:5]:  # Print first 5 files as example
                    print(f"{subindent}{f}")
                if len(files) > 5:
                    print(f"{subindent}...")

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
            print(f"Loading image from: {img_path}")
            
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

# Example usage
if __name__ == "__main__":
    # Replace with your actual dataset path
    dataset_path = r"C:\Users\GCCISAdmin\Downloads\group1-biometrics\NISTSpecialDatabase4GrayScaleImagesofFIGS"
    analyzer = BifurcationAnalyzer(dataset_path)
    
    try:
        # Load and compare a pair of fingerprints
        # Use just the base name of the file (without extension)
        img1 = analyzer.load_image("f0002")
        img2 = analyzer.load_image("s0002")
        
        # Extract features
        skeleton1, points1 = analyzer.extract_bifurcation_features(img1)
        skeleton2, points2 = analyzer.extract_bifurcation_features(img2)
        
        # Compare fingerprints
        similarity = analyzer.compare_fingerprints(points1, points2)
        print(f"Similarity score: {similarity:.3f}")
        
        # Visualize results
        analyzer.visualize_results(skeleton1, points1, "First Fingerprint")
        analyzer.visualize_results(skeleton2, points2, "Second Fingerprint")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")